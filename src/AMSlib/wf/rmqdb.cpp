/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <cstdint>

#include "wf/basedb.hpp"

using namespace ams::db;

/**
 * AMSMsgHeader
 */

AMSMsgHeader::AMSMsgHeader(size_t mpi_rank,
                           size_t domain_size,
                           size_t in_dim,
                           size_t out_dim)
    : hsize(static_cast<uint8_t>(AMSMsgHeader::size())),
      mpi_rank(static_cast<uint16_t>(mpi_rank)),
      domain_size(static_cast<uint16_t>(domain_size)),
      in_dim(static_cast<uint16_t>(in_dim)),
      out_dim(static_cast<uint16_t>(out_dim))
{
}

AMSMsgHeader::AMSMsgHeader(uint16_t mpi_rank,
                           uint16_t domain_size,
                           uint16_t in_dim,
                           uint16_t out_dim)
    : hsize(static_cast<uint8_t>(AMSMsgHeader::size())),
      mpi_rank(mpi_rank),
      domain_size(domain_size),
      in_dim(in_dim),
      out_dim(out_dim)
{
}

size_t AMSMsgHeader::encode(uint8_t* data_blob)
{
  if (!data_blob) return 0;

  size_t current_offset = 0;
  // MPI rank (should be 2 bytes)
  current_offset += serialize_data(&data_blob[current_offset], hsize);
  current_offset += serialize_data(&data_blob[current_offset], mpi_rank);
  current_offset += serialize_data(&data_blob[current_offset], domain_size);
  current_offset +=
      serialize_data(&data_blob[current_offset], static_cast<uint16_t>(in_dim));
  current_offset += serialize_data(&data_blob[current_offset],
                                   static_cast<uint16_t>(out_dim));

  // Domain Size (should be 2 bytes)
  AMS_DBG(AMSMsgHeader,
          "Generating domain name of size {} --- {}",
          domain_size,
          sizeof(domain_size));
  return AMSMsgHeader::size();
}

AMSMsgHeader AMSMsgHeader::decode(uint8_t* data_blob)
{
  size_t current_offset = 0;
  // Header size (should be 1 bytes)
  uint8_t new_hsize = data_blob[current_offset];
  AMS_CWARNING(AMSMsgHeader,
               new_hsize != AMSMsgHeader::size(),
               "buffer is likely not a valid AMSMessage ({} / {})",
               new_hsize,
               current_offset)

  current_offset += sizeof(uint8_t);
  // Data type (should be 1 bytes)
  uint8_t new_dtype = data_blob[current_offset];
  current_offset += sizeof(uint8_t);
  // MPI rank (should be 2 bytes)
  uint16_t new_mpirank =
      (reinterpret_cast<uint16_t*>(data_blob + current_offset))[0];
  current_offset += sizeof(uint16_t);

  // Domain Size (should be 2 bytes)
  uint16_t new_domain_size =
      (reinterpret_cast<uint16_t*>(data_blob + current_offset))[0];
  current_offset += sizeof(uint16_t);

  // Num elem (should be 4 bytes)
  uint32_t new_num_elem;
  std::memcpy(&new_num_elem, data_blob + current_offset, sizeof(uint32_t));
  current_offset += sizeof(uint32_t);
  // Input dim (should be 2 bytes)
  uint16_t new_in_dim;
  std::memcpy(&new_in_dim, data_blob + current_offset, sizeof(uint16_t));
  current_offset += sizeof(uint16_t);
  // Output dim (should be 2 bytes)
  uint16_t new_out_dim;
  std::memcpy(&new_out_dim, data_blob + current_offset, sizeof(uint16_t));

  return AMSMsgHeader(new_mpirank, new_domain_size, new_in_dim, new_out_dim);
}

/**
 * AMSMessage
 */

void AMSMessage::swap(const AMSMessage& other)
{
  _id = other._id;
  _rank = other._rank;
  _input_dim = other._input_dim;
  _output_dim = other._output_dim;
  _total_size = other._total_size;
  _data = other._data;
}


/**
 * AMSMessageInbound
 */

AMSMessageInbound::AMSMessageInbound(uint64_t id,
                                     uint64_t rId,
                                     std::string body,
                                     std::string exchange,
                                     std::string routing_key,
                                     bool redelivered)
    : id(id),
      rId(rId),
      body(std::move(body)),
      exchange(std::move(exchange)),
      routing_key(std::move(routing_key)),
      redelivered(redelivered) {};


bool AMSMessageInbound::empty() { return body.empty() || routing_key.empty(); }

bool AMSMessageInbound::isTraining()
{
  auto split = splitString(body, ":");
  return split[0] == "UPDATE";
}

std::string AMSMessageInbound::getModelPath()
{
  auto split = splitString(body, ":");
  if (split[0] == "UPDATE") {
    return split[1];
  }
  return {};
}

std::vector<std::string> AMSMessageInbound::splitString(std::string str,
                                                        std::string delimiter)
{
  size_t pos = 0;
  std::string token;
  std::vector<std::string> res;
  while ((pos = str.find(delimiter)) != std::string::npos) {
    token = str.substr(0, pos);
    res.push_back(token);
    str.erase(0, pos + delimiter.length());
  }
  res.push_back(str);
  return res;
}

/**
 * MessageQueue
 */

void MessageQueue::push(const PublishMessage& msg)
{
  std::lock_guard<std::mutex> lock(_mutex);
  _queue.push(msg);
}

bool MessageQueue::pop(PublishMessage& msg)
{
  std::lock_guard<std::mutex> lock(_mutex);
  if (_queue.empty()) return false;
  msg = _queue.front();
  _queue.pop();
  return true;
}

size_t MessageQueue::size()
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _queue.size();
}

/**
 * MessagesBuffer
 */

bool MessagesBuffer::insert(const PublishMessage& msg)
{
  std::lock_guard<std::mutex> lock(_mutex);
  if (_msgs.count(msg.id) == 1) return false;
  _msgs[msg.id] = msg;
  return true;
}

void MessagesBuffer::erase(int id)
{
  std::lock_guard<std::mutex> lock(_mutex);
  _msgs.erase(id);
}

void MessagesBuffer::print()
{
  std::lock_guard<std::mutex> lock(_mutex);
  for (const auto& e : _msgs)
    AMS_DBG(MessagesBuffer,
            "Message [{}] (addr={},use_count={}, size={})",
            e.second.id,
            static_cast<void*>(e.second.dPtr.get()),
            e.second.dPtr.use_count(),
            e.second.size);
}

size_t MessagesBuffer::size()
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _msgs.size();
}

/**
 * AMQPHandler
 */

void AMQPHandler::onDetached(AMQP::TcpConnection* connection)
{
  AMS_DBG(AMQPHandler, "Connection detached");
  // Signal reconnection if needed.
  if (reconnectCallback) reconnectCallback();
}

void AMQPHandler::onError(AMQP::TcpConnection* connection, const char* message)
{
  AMS_WARNING(AMQPHandler, "Connection error: '{}'", message)
  if (reconnectCallback) reconnectCallback();
}

bool AMQPHandler::onSecuring(AMQP::TcpConnection* connection, SSL* ssl)
{
  // No TLS certificate provided
  if (_cacert.empty()) {
    AMS_DBG(AMQPHandler, "No TLS certificate. Bypassing.")
    return true;
  }

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
    AMS_WARNING(AMQPHandler, "{}", error)
    return false;
  } else {
    AMS_DBG(AMQPHandler, "Success logged with ca-chain")
    return true;
  }
}

bool AMQPHandler::onSecured(AMQP::TcpConnection* connection, const SSL* ssl)
{
  AMS_DBG(AMQPHandler, "Secured TLS connection has been established")
  return true;
}

void AMQPHandler::onClosed(AMQP::TcpConnection* connection)
{
  AMS_DBG(AMQPHandler, "Connection closed")
}

void AMQPHandler::onReady(AMQP::TcpConnection* connection)
{
  AMS_DBG(AMQPHandler, "Connection established and ready")
}
