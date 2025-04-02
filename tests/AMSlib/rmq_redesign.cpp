#ifdef __AMS_ENABLE_MPI__
#include <mpi.h>
#endif

#include <AMS.h>
#include <amqpcpp.h>
#include <amqpcpp/libevent.h>
#include <amqpcpp/linux_tcp.h>
#include <amqpcpp/reliable.h>
#include <amqpcpp/throttle.h>
#include <event2/event-config.h>
#include <event2/event.h>
#include <event2/thread.h>
#include <openssl/err.h>
#include <openssl/opensslv.h>
#include <openssl/ssl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>
#include <thread>
#include <unordered_map>


#include "wf/basedb.hpp"
using json = nlohmann::json;

// Structure to hold a publish request.
struct PublishMessage {
  std::shared_ptr<uint8_t> dPtr;
  size_t size;
  int id;
  PublishMessage() : dPtr(nullptr), size(-1), id(-1) {}
  PublishMessage(std::shared_ptr<uint8_t>& dPtr, size_t size, int id)
      : dPtr(dPtr), size(size), id(id)
  {
  }

  // TODO: implement some move semantics to avoid copying shared_ptr (expensive)
};

class MessagesBuffer;

// A simple thread-safe queue for publish messages.
class MessageQueue
{
public:
  void push(const PublishMessage& msg)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _queue.push(msg);
  }

  // Returns true if a message was popped.
  bool pop(PublishMessage& msg)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_queue.empty()) return false;
    msg = _queue.front();
    _queue.pop();
    return true;
  }

  // Returns size of the queue
  size_t size()
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _queue.size();
  }

private:
  std::queue<PublishMessage> _queue;
  std::mutex _mutex;
};



class MessagesBuffer {
private:
  using iterator_t = std::unordered_map<int, PublishMessage>::iterator;
  // Note: we could remove the id from PublishMessage struct
  std::unordered_map<int, PublishMessage> _msgs;
  std::shared_mutex _mutex;

  MessagesBuffer() = default;

public:
  MessagesBuffer(MessagesBuffer&) = delete;
  MessagesBuffer& operator=(MessagesBuffer&) = delete;

  MessagesBuffer(MessagesBuffer&&) = delete;
  MessagesBuffer& operator=(MessagesBuffer&&) = delete;

  iterator_t begin() { return std::begin(_msgs); }

  iterator_t end() { return std::end(_msgs); }

  void insert(const PublishMessage& msg) {
    std::unique_lock<std::shared_mutex> lock(_mutex);
    _msgs[msg.id] = msg;
  }

  void erase(int id) {
    std::unique_lock<std::shared_mutex> lock(_mutex);
    iterator_t it = _msgs.find(id);
    if (it != end()) {
      std::cout << "Erasing msg " << id << std::endl;
      _msgs.erase(it);
    }
  }

  void print()
  {
    std::shared_lock<std::shared_mutex> lock(_mutex);
    for (const auto& e : _msgs)
      fprintf(stdout,
          "Message [%d] (addr=%p,use_count=%d, size=%d)\n",
          e.second.id,
          e.second.dPtr.get(),
          e.second.dPtr.use_count(),
          e.second.size);
  }

  size_t size() {
      std::shared_lock<std::shared_mutex> lock(_mutex);
      return _msgs.size();
  }

  static MessagesBuffer& getInstance() {
    static MessagesBuffer instance;
    return instance;
  }
};

// Custom handler for AMQP events.
class MyAMQPHandler : public AMQP::LibEventHandler
{
public:
  MyAMQPHandler(struct event_base* base, std::string cacert)
      : AMQP::LibEventHandler(base), _cacert(cacert)
  {
  }

  virtual void onDetached(AMQP::TcpConnection* connection) override
  {
    std::cerr << "MyAMQPHandler: Connection detached." << std::endl;
    // Signal reconnection if needed.
    if (reconnectCallback) reconnectCallback();
  }

  virtual void onError(AMQP::TcpConnection* connection,
                       const char* message) override
  {
    std::cerr << "MyAMQPHandler: Connection error: " << message << std::endl;
    if (reconnectCallback) reconnectCallback();
  }

  virtual bool onSecuring(AMQP::TcpConnection* connection, SSL* ssl) override
  {
    // No TLS certificate provided
    ERR_clear_error();
    unsigned long err;
#if OPENSSL_VERSION_NUMBER < 0x10100000L
    int ret = SSL_use_certificate_file(ssl, _cacert.c_str(), SSL_FILETYPE_PEM);
#else
    int ret = SSL_use_certificate_chain_file(ssl, _cacert.c_str());
#endif
    if (ret != 1) {
      std::string error("openssl: error loading ca-chain () + from [");
      SSL_get_error(ssl, ret);
      if ((err = ERR_get_error())) {
        error += std::string(ERR_reason_error_string(err));
      }
      error += "]";
      return false;
    } else {
      std::cout << "Success logged with ca-chain \n";
      return true;
    }
  }

  virtual bool onSecured(AMQP::TcpConnection* connection,
                         const SSL* ssl) override
  {
    std::cout << "Secured \n";
    return true;
  }

  virtual void onClosed(AMQP::TcpConnection* connection) override
  {
    std::cout << "On closed \n";
  }

  virtual void onReady(AMQP::TcpConnection* connection) override
  {
    std::cout << "Connection established and ready." << std::endl;
  }


  // Set by the connection manager.
  std::function<void()> reconnectCallback;
  std::string _cacert;
};

class ConnectionManager
{
public:
  ConnectionManager(std::string rmq_password,
                    std::string rmq_user,
                    std::string rmq_vhost,
                    int service_port,
                    std::string service_host,
                    std::string rmq_cert,
                    std::string outbound_queue,
                    std::string exchange,
                    std::string routing_key)
      : _address(service_host,
                 service_port,
                 AMQP::Login(rmq_user, rmq_password),
                 rmq_vhost,
                 rmq_cert.empty() ? false : true),
        _stop(false),
        _queue_sender(outbound_queue),
        _exchange(exchange),
        _routing_key(routing_key),
        _reconnecting(false)
  {
    std::cout << _address << "\n";
    evthread_use_pthreads();

    _base = event_base_new();
    _handler = std::make_shared<MyAMQPHandler>(_base, rmq_cert);
    // Set up the reconnection callback.
    _handler->reconnectCallback =
        std::bind(&ConnectionManager::scheduleReconnect, this);

    _sendEvent = event_new(
        _base, -1, EV_PERSIST, ConnectionManager::sendMessageCallback, this);
    // Add the event (without a timeout, since we trigger it manually).
    event_add(_sendEvent, nullptr);
    std::cout << "Event callback " << _sendEvent << "\n";

    // 2000ms timer to simulate connection drops
    struct timeval tv = {2, 0};  // Every 2 seconds
    _dropConnectionEvent = event_new(
        _base, -1, EV_PERSIST, ConnectionManager::simulateConnectionDrop, this);
    event_add(_dropConnectionEvent, &tv);  // Add the event to the event loop

    // Start the worker thread.
    createConnection();
    _workerThread = std::thread([this]() {
      std::cout << "Starting dispatch \n";
      event_base_dispatch(_base);
      std::cout << "Stop dispatch\n";
    });
  }

  ~ConnectionManager()
  {
    if (!_stop) stop();
    std::cout << "Stopped\n";
    if (_workerThread.joinable()) _workerThread.join();
    std::cout << "Joined \n";
    if (_dropConnectionEvent) event_free(_dropConnectionEvent);
    if (_sendEvent) event_free(_sendEvent);
    if (_base) event_base_free(_base);
  }


  void publish(const PublishMessage& msg) {
    std::cout << "Setting send event callback " << _sendEvent << "\n";
    _msgQueue.push(msg);
    event_active(this->_sendEvent, EV_WRITE, 0);
  }

  void closeConnection()
  {
    if (_connection) {
      close(_connection->fileno());  // Close the connection
      std::cout << "Connection closed." << std::endl;
    }
  }

  static void simulateConnectionDrop(evutil_socket_t, short, void* arg)
  {
    ConnectionManager* mgr = reinterpret_cast<ConnectionManager*>(arg);
    std::cout << "Simulating connection drop..." << std::endl;

    // Close the current connection (simulate a drop)
    mgr->closeConnection();
  }

  // Signal shutdown.
  void stop()
  {
    std::cerr << "we have " << MessagesBuffer::getInstance().size() << " messages not acked\n";

    // We want to have a chance to send the messages that could not be sent before
    if (MessagesBuffer::getInstance().size() > 0) {
      std::cout << "We try to resend messages before stopping\n";
      // Here if we cannot call directly processQueue() as it will deadlock (waiting on lock in event_del_)
      for (auto& item : MessagesBuffer::getInstance())
        _msgQueue.push(item.second);

      int iters = 0;
      int repeat = 10;
      while ((MessagesBuffer::getInstance().size() != 0) && (iters++ < repeat)) {
        std::cerr << "Waiting " << iters << "/" << repeat << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }

    _stop = true;
    event_base_loopexit(_base, nullptr);
  }

private:
  struct event_base* _base;
  struct event* _sendEvent;
  struct event* _dropConnectionEvent;
  std::shared_ptr<MyAMQPHandler> _handler;
  AMQP::Address _address;
  std::unique_ptr<AMQP::TcpConnection> _connection;
  // Use both a plain channel and a reliable wrapper for confirmations.
  std::shared_ptr<AMQP::TcpChannel> _channel;
  std::shared_ptr<AMQP::Reliable<AMQP::Tagger>> _reliableChannel;
  MessageQueue _msgQueue;
  std::thread _workerThread;
  std::atomic<bool> _stop;
  std::atomic<bool> _reconnecting;
  std::string _queue_sender;
  std::string _exchange;
  std::string _routing_key;

  void internalPublish(const PublishMessage& msg) {
    // Publish using the reliable channel if available.
    if (_reliableChannel) {
      _reliableChannel
          ->publish("",
                    _queue_sender,
                    reinterpret_cast<char*>(msg.dPtr.get()),
                    msg.size)
          .onAck([msg]() {
            std::cout << "Ok: Message acked: " << msg.id << std::endl;
            // If msg is in the MessagesBuffer, we erase it
            MessagesBuffer::getInstance().erase(msg.id);
          })
          .onNack([this, msg]() {
            std::cerr << "Warning: Message nack'ed, requeueing: " << msg.id
                      << std::endl;
            MessagesBuffer::getInstance().insert(msg);
          })
          .onError([this, msg](const char* errMsg) {
            std::cerr << "Error: Publish error for message (" << msg.id
                      << "): " << errMsg << std::endl;
            MessagesBuffer::getInstance().insert(msg);
          });
    } else {
      std::cerr << "No valid channel for publishing." << std::endl;
      MessagesBuffer::getInstance().insert(msg);
    }
  }

  void processQueue()
  {
    // Publishing the current msgs buffered
    PublishMessage msg;
    while (_msgQueue.size() > 0) {
      if (_msgQueue.pop(msg)) {
        std::cout << "Processing message: " << msg.id << std::endl;
        internalPublish(msg);
      }
    }

    // Re-publishing messages that have not been ack-ed
    for (auto& item : MessagesBuffer::getInstance()) {
      std::cout << "re-processing message: " << item.first << std::endl;
      internalPublish(item.second);
    }
  }

  // The send event callback. This is called when _sendEvent is activated.
  static void sendMessageCallback(evutil_socket_t, short, void* arg)
  {
    std::cout << "Sending message callback\n";
    ConnectionManager* mgr = reinterpret_cast<ConnectionManager*>(arg);
    mgr->processQueue();
  }

  // Create connection, channel, and wrap the channel in a reliable channel.
  void createConnection()
  {
    _connection =
        std::make_unique<AMQP::TcpConnection>(_handler.get(), _address);
    _channel = std::make_shared<AMQP::TcpChannel>(_connection.get());
    // Wrap the plain channel in a reliable channel for publish confirmations.
    _reliableChannel =
        std::make_shared<AMQP::Reliable<AMQP::Tagger>>(*_channel);
    std::cout << "Connection and channels established." << std::endl;
  }

  // Schedule a reconnect if not already in progress.
  void scheduleReconnect()
  {
    if (_stop) return;
    if (_reconnecting.exchange(true)) return;  // Already reconnecting.
    struct timeval tv = {0, 0};                // 1 second delay.
    event_base_once(_base,
                    -1,
                    EV_TIMEOUT,
                    ConnectionManager::reconnectTimerCallback,
                    this,
                    &tv);
  }

  // Static callback wrapper for the timer.
  static void reconnectTimerCallback(evutil_socket_t, short, void* arg)
  {
    ConnectionManager* mgr = static_cast<ConnectionManager*>(arg);
    mgr->reconnect();
  }

  // Reconnect by closing the old connection and re-creating everything.
  void reconnect()
  {
    std::cout << "Reconnecting..." << std::endl;
    if (_connection) {
      _connection->close();
    }
    // Clean up the channels.
    _channel.reset();
    _reliableChannel.reset();
    createConnection();
    _reconnecting = false;
    std::cout << "Reconnection complete." << std::endl;
  }
};

template <typename T>
T getEntry(json& entry, std::string field)
{
  if (!entry.contains(field)) {
    throw std::runtime_error("I was expecting entry '" + field +
                             "' to exist in json");
  }
  return entry[field].get<T>();
}


//
// Example usage:
//
int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::cout << "Wrong cli, expected one: \n";
    std::cout << argv[0] << " <json-ams-config> <num-messages>\n";
    return -1;
  }

  // Number of ranks
  int wS = 1;
  // My Local Id
  int rId = 0;
  // Level of Threading provided by MPI
  int provided = 0;
  MPI_CALL(MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided));
  MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &wS));
  MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rId));

  auto& mg = ams::ResourceManager::getInstance();
  mg.init();
  auto json_file = std::string(argv[1]);
  int num_messages = std::atoi(argv[2]);
  std::cout << "Reading " << json_file << "\n";
  std::ifstream json_stream(json_file);
  json jRoot = json::parse(json_stream);
  if (!jRoot.contains("db")) {
    throw std::runtime_error("expecting db field in json");
    return -1;
  }
  auto entry = jRoot["db"];
  if (!entry.contains("dbType"))
    std::runtime_error(
        "JSON file instantiates db-fields without defining a "
        "\"dbType\" "
        "entry");
  auto dbStrType = entry["dbType"].get<std::string>();


  if (!entry.contains("rmq_config")) {
    throw std::runtime_error(
        "JSON db-fields do not contain rmq_config entires");
  }
  auto rmq_entry = entry["rmq_config"];
  int port = getEntry<int>(rmq_entry, "service-port");
  std::string host = getEntry<std::string>(rmq_entry, "service-host");
  std::string rmq_pass = getEntry<std::string>(rmq_entry, "rabbitmq-password");
  std::string rmq_user = getEntry<std::string>(rmq_entry, "rabbitmq-user");
  std::string rmq_vhost = getEntry<std::string>(rmq_entry, "rabbitmq-vhost");
  std::string rmq_out_queue =
      getEntry<std::string>(rmq_entry, "rabbitmq-queue-physics");
  std::string exchange =
      getEntry<std::string>(rmq_entry, "rabbitmq-exchange-training");
  std::string routing_key =
      getEntry<std::string>(rmq_entry, "rabbitmq-key-training");
  // Each MPI rank would create its own ConnectionManager instance.
  std::string rmq_cert = "";
  if (rmq_entry.contains("rabbitmq-cert"))
    rmq_cert = getEntry<std::string>(rmq_entry, "rabbitmq-cert");


  ConnectionManager connManager(rmq_pass,
                                rmq_user,
                                rmq_vhost,
                                port,
                                host,
                                rmq_cert,
                                rmq_out_queue,
                                exchange,
                                routing_key);

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  // Simulate publishing messages from the main thread.
  int num_inputs = 5;
  int num_outputs = 10;
  size_t size = 1000;
  std::string domain_name("test");
  for (int i = 0; i < num_messages; i++) {
    std::cout << "[" << rId << " / " << wS << "] Pushing message " << i << "\n";
    std::vector<double*> Inputs;
    std::vector<double*> Outputs;
    for (int i = 0; i < num_inputs; i++)
      Inputs.push_back((double*)malloc(size * sizeof(double)));
    for (int i = 0; i < num_outputs; i++) {
      Outputs.push_back((double*)malloc(sizeof(double) * size));
    }

    ams::db::AMSMessage msg(i,
                            rId,
                            domain_name,
                            1000,
                            Inputs,
                            Outputs);

    std::shared_ptr<uint8_t> ptr(msg.data(),
                                 ams::db::AMSMessageRecords::getDeleter());
    PublishMessage record(ptr, msg.size(), i);
    connManager.publish(record);
    for (auto& I : Inputs)
      free(I);
    for (auto& O : Outputs)
      free(O);
    if (i % 5) // to slow down the process enough to let the simulated failures be impactul
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  connManager.stop();

  int global_sum = 0;
  int local_sum = MessagesBuffer::getInstance().size();
  #ifdef __AMS_ENABLE_MPI__
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  #else
  global_sum = local_sum;
  #endif

  if (rId == 0) {
    std::cout << "Total number of messages that should have been sent : " << wS * num_messages << "\n";
    std::cout << "Total number of messages acked : " << (wS*num_messages-global_sum) << "\n";
    std::cout << "Total number of messages non-acked : " << global_sum << "\n";
  } else {
    std::cout << "Rank = " << rId << " number of messages that should have been sent : " << num_messages << "\n";
    std::cout << "Rank = " << rId << " number of messages acked : " << (num_messages-local_sum) << "\n";
    std::cout << "Rank = " << rId << " number of messages non-acked : " << local_sum << "\n";
  }
  
  MPI_CALL(MPI_Finalize());

  return 0;
}
