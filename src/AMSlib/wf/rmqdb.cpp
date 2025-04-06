/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "wf/basedb.hpp"

using namespace ams::db;

/**
 * AMSMsgHeader
 */

AMSMsgHeader::AMSMsgHeader(size_t mpi_rank,
                           size_t domain_size,
                           size_t num_elem,
                           size_t in_dim,
                           size_t out_dim,
                           size_t type_size)
    : hsize(static_cast<uint8_t>(AMSMsgHeader::size())),
      dtype(static_cast<uint8_t>(type_size)),
      mpi_rank(static_cast<uint16_t>(mpi_rank)),
      domain_size(static_cast<uint16_t>(domain_size)),
      num_elem(static_cast<uint32_t>(num_elem)),
      in_dim(static_cast<uint16_t>(in_dim)),
      out_dim(static_cast<uint16_t>(out_dim))
{
}

AMSMsgHeader::AMSMsgHeader(uint16_t mpi_rank,
                           uint16_t domain_size,
                           uint32_t num_elem,
                           uint16_t in_dim,
                           uint16_t out_dim,
                           uint8_t type_size)
    : hsize(static_cast<uint8_t>(AMSMsgHeader::size())),
      dtype(type_size),
      mpi_rank(mpi_rank),
      domain_size(domain_size),
      num_elem(num_elem),
      in_dim(in_dim),
      out_dim(out_dim)
{
}

size_t AMSMsgHeader::encode(uint8_t* data_blob)
{
  if (!data_blob) return 0;

  size_t current_offset = 0;
  // Header size (should be 1 bytes)
  data_blob[current_offset] = hsize;
  current_offset += sizeof(hsize);
  // Data type (should be 1 bytes)
  data_blob[current_offset] = dtype;
  current_offset += sizeof(dtype);
  // MPI rank (should be 2 bytes)
  std::memcpy(data_blob + current_offset, &(mpi_rank), sizeof(mpi_rank));
  current_offset += sizeof(mpi_rank);
  // Domain Size (should be 2 bytes)
  DBG(AMSMsgHeader,
      "Generating domain name of size %d --- %d offset %d",
      domain_size,
      sizeof(domain_size),
      current_offset);
  std::memcpy(data_blob + current_offset, &(domain_size), sizeof(domain_size));
  current_offset += sizeof(domain_size);
  // Num elem (should be 4 bytes)
  std::memcpy(data_blob + current_offset, &(num_elem), sizeof(num_elem));
  current_offset += sizeof(num_elem);
  // Input dim (should be 2 bytes)
  std::memcpy(data_blob + current_offset, &(in_dim), sizeof(in_dim));
  current_offset += sizeof(in_dim);
  // Output dim (should be 2 bytes)
  std::memcpy(data_blob + current_offset, &(out_dim), sizeof(out_dim));
  current_offset += sizeof(out_dim);

  return AMSMsgHeader::size();
}

AMSMsgHeader AMSMsgHeader::decode(uint8_t* data_blob)
{
  size_t current_offset = 0;
  // Header size (should be 1 bytes)
  uint8_t new_hsize = data_blob[current_offset];
  CWARNING(AMSMsgHeader,
           new_hsize != AMSMsgHeader::size(),
           "buffer is likely not a valid AMSMessage (%d / %ld)",
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

  return AMSMsgHeader(new_mpirank,
                      new_domain_size,
                      new_num_elem,
                      new_in_dim,
                      new_out_dim,
                      new_dtype);
}

/**
 * AMSMessage
 */

void AMSMessage::swap(const AMSMessage& other)
{
  _id = other._id;
  _rank = other._rank;
  _num_elements = other._num_elements;
  _input_dim = other._input_dim;
  _output_dim = other._output_dim;
  _total_size = other._total_size;
  _data = other._data;
}

AMSMessage::AMSMessage(int id, uint64_t rId, uint8_t* data)
    : _id(id),
      _num_elements(0),
      _input_dim(0),
      _output_dim(0),
      _data(data),
      _total_size(0)
{
  auto header = AMSMsgHeader::decode(data);

  int current_rank = rId;
  _rank = header.mpi_rank;
  CWARNING(AMSMessage,
           _rank != current_rank,
           "MPI rank are not matching (using %d)",
           _rank)

  _num_elements = header.num_elem;
  _input_dim = header.in_dim;
  _output_dim = header.out_dim;
  _data = data;
  auto type_value = header.dtype;

  _total_size = AMSMsgHeader::size() + getTotalElements() * type_value;

  DBG(AMSMessage, "Allocated message %d: %p", _id, _data);
}



bool AMQPHandler::waitToConnect(const std::chrono::milliseconds& duration) {
  auto lock = std::unique_lock<std::mutex>(_mutex);
  return _cv.wait_for(lock, duration, [&]() { return _status == ConnectionStatus::CONNECTED;});
}

void AMQPHandler::onDetached(AMQP::TcpConnection* connection)
{
  DBG(AMQPHandler, "Connection detached");
  // Signal reconnection if needed.
  if (reconnectCallback) reconnectCallback();
}

void AMQPHandler::onError(AMQP::TcpConnection* connection,
                      const char* message)
{
  WARNING(AMQPHandler, "Connection error: \"%s\"", message)
  if (reconnectCallback) reconnectCallback();
}

bool AMQPHandler::onSecuring(AMQP::TcpConnection* connection, SSL* ssl)
{
  // No TLS certificate provided
  if (_cacert.empty()) {
    DBG(AMQPHandler, "No TLS certificate. Bypassing.")
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
    CFATAL(AMQPHandler, false, "%s", error.c_str())
    return false;
  } else {
    DBG(AMQPHandler, "Success logged with ca-chain")
    return true;
  }
}

bool AMQPHandler::onSecured(AMQP::TcpConnection* connection,
                        const SSL* ssl)
{
  DBG(AMQPHandler, "Secured TLS connection has been established")
  return true;
}

void AMQPHandler::onClosed(AMQP::TcpConnection* connection)
{
  DBG(AMQPHandler, "Connection closed")
}

void AMQPHandler::onReady(AMQP::TcpConnection* connection)
{
  DBG(AMQPHandler, "Connection established and ready")
  {
    auto lock = std::lock_guard<std::mutex>(_mutex);
    _status = ConnectionStatus::CONNECTED;
  }
  _cv.notify_one();
}