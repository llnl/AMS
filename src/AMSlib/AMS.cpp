/*
 * Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
 * AMSLib Project Developers
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include <limits.h>

#ifdef __AMS_ENABLE_MPI__
#include <mpi.h>
#endif
#include <unistd.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "AMS.h"
#include "wf/basedb.hpp"
#include "wf/debug.h"
#include "wf/logger.hpp"
#include "wf/resource_manager.hpp"
#include "wf/workflow.hpp"

using namespace ams;

namespace
{

struct AMSAbstractModel {
  enum UQAggrType {
    Unknown = -1,
    Mean = 0,
    Max = 1,
  };

public:
  std::string SPath;
  std::string DBLabel;
  bool DebugDB;
  double threshold;
  AMSUQPolicy uqPolicy;

  static AMSUQPolicy getUQType(std::string type)
  {
    if (type.compare("deltaUQ") == 0) {
      return AMSUQPolicy::AMS_DELTAUQ_MEAN;
    } else if (type.compare("random") == 0) {
      return AMSUQPolicy::AMS_RANDOM;
    } else {
      THROW(std::runtime_error, "Unknown uq type " + type);
    }
    return AMSUQPolicy::AMS_UQ_END;
  }

  static UQAggrType getUQAggregate(std::string policy)
  {
    if (policy.compare("mean") == 0)
      return UQAggrType::Mean;
    else if (policy.compare("max") == 0)
      return UQAggrType::Max;
    return UQAggrType::Unknown;
  }

  std::string parseDBLabel(nlohmann::json &value)
  {
    if (!value.contains("db_label")) {
      THROW(std::runtime_error, "ml model must contain <db_label> entry");
    }

    return value["db_label"].get<std::string>();
  }

  bool parseDebugDB(nlohmann::json &value)
  {
    if (!value.contains("debug_db")) {
      return false;
    }

    return value["debug_db"].get<bool>();
  }


  std::string parseSurrogatePaths(nlohmann::json &jRoot)
  {

    std::string path = "";
    if (jRoot.contains("model_path")) {
      path = jRoot["model_path"].get<std::string>();
      CFATAL(AMS,
             (!path.empty() && !fs::exists(path)),
             "Path '%s' to model does not exist\n",
             path.c_str());
    }
    return path;
  }


  int parseClusters(nlohmann::json &value)
  {
    if (!value.contains("neighbours"))
      THROW(std::runtime_error, "UQ Policy must contain neighbours");

    return value["neighbours"].get<int>();
  }


  AMSUQPolicy parseUQPolicy(nlohmann::json &value)
  {
    AMSUQPolicy policy = AMSUQPolicy::AMS_UQ_END;
    if (value.contains("uq_type")) {
      policy = getUQType(value["uq_type"].get<std::string>());
    } else {
      THROW(std::runtime_error, "Model must specify the UQ type");
    }

    if (!UQ::isUQPolicy(policy)) {
      THROW(std::runtime_error, "UQ Policy is not supported");
    }

    UQAggrType uqAggregate = UQAggrType::Unknown;
    if (value.contains("uq_aggregate")) {
      uqAggregate = getUQAggregate(value["uq_aggregate"].get<std::string>());
    } else {
      THROW(std::runtime_error, "Model must specify UQ Policy");
    }


    if (UQ::isDeltaUQ(policy) && uqAggregate == UQAggrType::Unknown) {
      THROW(std::runtime_error,
            "UQ Type should be defined or set to undefined value");
    }

    if (UQ::isDeltaUQ(policy)) {
      if (uqAggregate == Max)
        policy = AMSUQPolicy::AMS_DELTAUQ_MAX;
      else if (uqAggregate == Mean)
        policy = AMSUQPolicy::AMS_DELTAUQ_MEAN;
    }
    DBG(AMS, "UQ Policy is %s", UQ::UQPolicyToStr(policy).c_str())
    return policy;
  }


public:
  AMSAbstractModel(nlohmann::json &value)
  {

    uqPolicy = parseUQPolicy(value);

    if (!value.contains("threshold")) {
      THROW(std::runtime_error,
            "Model must define threshold value (threshold < 0 always "
            "performs original code, threshold=1e30 always use the "
            "model)");
    }
    threshold = value["threshold"].get<float>();
    SPath = parseSurrogatePaths(value);
    DBLabel = parseDBLabel(value);
    DebugDB = parseDebugDB(value);

    CFATAL(AMS,
           DebugDB && (SPath.empty()),
           "To store predicates in dabase, a surrogate model field is "
           "mandatory");
  }


  AMSAbstractModel(AMSUQPolicy uq_policy,
                   const char *surrogate_path,
                   const char *db_label,
                   double threshold)
  {
    DebugDB = false;
    if (db_label == nullptr)
      FATAL(AMS, "registering model without a database identifier\n");

    DBLabel = std::string(db_label);

    if (!UQ::isUQPolicy(uq_policy)) {
      FATAL(AMS, "Invalid UQ policy %d", uq_policy)
    }

    uqPolicy = uq_policy;

    if (surrogate_path != nullptr) SPath = std::string(surrogate_path);

    this->threshold = threshold;
    DBG(AMS,
        "Registered Model %s %g",
        UQ::UQPolicyToStr(uqPolicy).c_str(),
        threshold);
  }


  void dump()
  {
    if (!SPath.empty()) DBG(AMS, "Surrogate Model Path: %s", SPath.c_str());
    DBG(AMS,
        "db-Label: %s threshold %f UQ-Policy: %u",
        DBLabel.c_str(),
        threshold,
        uqPolicy);
  }
};


/* The class is reponsible to instantiate and 
 * initialize objects from environment variables
 * and acts as the C to CPP wrapper
 */
class AMSWrap
{
  using json = nlohmann::json;

public:
  std::vector<void *> executors;
  std::vector<std::pair<std::string, AMSAbstractModel>> registered_models;
  std::unordered_map<std::string, int> ams_candidate_models;
  AMSDBType dbType = AMSDBType::AMS_NONE;
  ams::ResourceManager &memManager;
  int rId;

private:
  void dumpEnv()
  {
    for (auto &KV : ams_candidate_models) {
      DBG(AMS, "\n")
      DBG(AMS,
          "\t\t\t Model: %s With AMSAbstractID: %d",
          KV.first.c_str(),
          KV.second);
      if (KV.second >= ams_candidate_models.size()) {
        FATAL(AMS,
              "Candidate model mapped to AMSAbstractID that does not exist "
              "(%d)",
              KV.second);
      }
      auto &abstract_model = registered_models[KV.second].second;
      abstract_model.dump();
    }
  }

  void parseDomainModels(
      json &jRoot,
      std::unordered_map<std::string, std::string> &domain_map)
  {
    if (!jRoot.contains("domain_models")) return;

    auto domain_models = jRoot["domain_models"];
    for (auto &field : domain_models.items()) {
      auto &name = field.key();
      auto val = field.value().get<std::string>();
      domain_map.emplace(val, name);
    }
    return;
  }

  void parseCandidateAMSModels(
      json &jRoot,
      std::unordered_map<std::string, std::string> ml_domain_mapping)
  {
    if (!jRoot.contains("ml_models")) return;
    auto models = jRoot["ml_models"];
    for (auto &field : models.items()) {
      // We skip models not registered to respective domains. We will not use
      // those.
      auto &key = field.key();
      if (ml_domain_mapping.find(key) == ml_domain_mapping.end()) continue;

      if (ams_candidate_models.find(ml_domain_mapping[key]) !=
          ams_candidate_models.end()) {
        FATAL(AMS,
              "Domain Model %s has multiple ml model mappings",
              ml_domain_mapping[key].c_str())
      }

      registered_models.push_back(
          std::make_pair(ml_domain_mapping[key],
                         AMSAbstractModel(field.value())));
      // We add the value of the domain mappings, as the application can
      // only query based on these.
      ams_candidate_models.emplace(ml_domain_mapping[key],
                                   registered_models.size() - 1);
    }
  }

  void setupFSDB(json &entry, std::string &dbStrType)
  {
    if (!entry.contains("fs_path"))
      THROW(std::runtime_error,
            "JSON db-fields does not provide file system path");

    std::string db_path = entry["fs_path"].get<std::string>();
    auto &DB = ams::db::DBManager::getInstance();
    DB.instantiate_fs_db(dbType, db_path);
    DBG(AMS,
        "Configured AMS File system database to point to %s using file "
        "type %s",
        db_path.c_str(),
        dbStrType.c_str());
  }

  template <typename T>
  T getEntry(json &entry, std::string field)
  {
    if (!entry.contains(field)) {
      THROW(std::runtime_error,
            ("I was expecting entry '" + field + "' to exist in json").c_str())
    }
    return entry[field].get<T>();
  }

  void setupRMQ(json &entry, std::string &dbStrType)
  {
    if (!entry.contains("rmq_config")) {
      THROW(std::runtime_error,
            "JSON db-fields do not contain rmq_config entires")
    }
    auto rmq_entry = entry["rmq_config"];
    int port = getEntry<int>(rmq_entry, "service-port");
    std::string host = getEntry<std::string>(rmq_entry, "service-host");
    std::string rmq_pass =
        getEntry<std::string>(rmq_entry, "rabbitmq-password");
    std::string rmq_user = getEntry<std::string>(rmq_entry, "rabbitmq-user");
    std::string rmq_vhost = getEntry<std::string>(rmq_entry, "rabbitmq-vhost");
    std::string rmq_out_queue =
        getEntry<std::string>(rmq_entry, "rabbitmq-queue-physics");
    std::string exchange =
        getEntry<std::string>(rmq_entry, "rabbitmq-exchange-training");
    std::string routing_key =
        getEntry<std::string>(rmq_entry, "rabbitmq-key-training");
    bool update_surrogate = getEntry<bool>(entry, "update_surrogate");

    // We allow connection to RabbitMQ without TLS certificate
    std::string rmq_cert = "";
    if (rmq_entry.contains("rabbitmq-cert"))
      rmq_cert = getEntry<std::string>(rmq_entry, "rabbitmq-cert");

    CFATAL(AMS,
           (exchange == "" || routing_key == "") && update_surrogate,
           "Found empty RMQ exchange / routing-key, model update is not "
           "possible. "
           "Please provide a RMQ exchange or deactivate surrogate model "
           "update.")

    if (exchange == "" || routing_key == "") {
      WARNING(AMS,
              "Found empty RMQ exchange or routing-key, deactivating model "
              "update")
      update_surrogate = false;
    }

    auto &DB = ams::db::DBManager::getInstance();
    DB.instantiate_rmq_db(port,
                          host,
                          rmq_pass,
                          rmq_user,
                          rmq_vhost,
                          rmq_cert,
                          rmq_out_queue,
                          exchange,
                          routing_key,
                          update_surrogate);
  }

  void parseDatabase(json &jRoot)
  {
    DBG(AMS, "Parsing Data Base Fields")
    if (!jRoot.contains("db")) return;
    auto entry = jRoot["db"];
    if (!entry.contains("dbType"))
      THROW(std::runtime_error,
            "JSON file instantiates db-fields without defining a "
            "\"dbType\" "
            "entry");
    auto dbStrType = entry["dbType"].get<std::string>();
    dbType = ams::db::getDBType(dbStrType);
    switch (dbType) {
      case AMSDBType::AMS_NONE:
        return;
      case AMSDBType::AMS_HDF5:
        setupFSDB(entry, dbStrType);
        break;
      case AMSDBType::AMS_RMQ:
        setupRMQ(entry, dbStrType);
        break;
      default:
        FATAL(AMS, "Unknown db-type");
    }
    return;
  }

public:
  AMSWrap() : memManager(ams::ResourceManager::getInstance())
  {
    memManager.init();

    if (const char *object_descr = std::getenv("AMS_OBJECTS")) {
      DBG(AMS, "Opening env file %s", object_descr);
      std::ifstream json_file(object_descr);
      json data = json::parse(json_file);
      /* We first parse domain models. Domain models can be potentially 
       * queried and returned to the main application using the "key" value
       * as query parameter. This redirection only applies for ml-models 
       * registered by the application itself.
       */
      std::unordered_map<std::string, std::string> domain_mapping;
      parseDomainModels(data, domain_mapping);
      parseCandidateAMSModels(data, domain_mapping);
      parseDatabase(data);
    }

    dumpEnv();
  }

  int register_model(const char *domain_name,
                     AMSUQPolicy uq_policy,
                     double threshold,
                     const char *surrogate_path,
                     const char *db_label)
  {
    auto model = ams_candidate_models.find(domain_name);
    if (model != ams_candidate_models.end()) {
      FATAL(AMS,
            "Trying to register model on domain: %s but model already exists "
            "%s",
            domain_name,
            registered_models[model->second].second.SPath.c_str());
    }
    registered_models.push_back(std::make_pair(
        std::string(domain_name),
        AMSAbstractModel(uq_policy, surrogate_path, db_label, threshold)));
    ams_candidate_models.emplace(std::string(domain_name),
                                 registered_models.size() - 1);
    return registered_models.size() - 1;
  }

  int get_model_index(const char *domain_name)
  {
    auto model = ams_candidate_models.find(domain_name);
    if (model == ams_candidate_models.end()) return -1;

    return model->second;
  }

  std::pair<std::string, AMSAbstractModel> &get_model(int index)
  {
    if (index >= registered_models.size()) {
      FATAL(AMS, "Model id: %d does not exist", index);
    }

    return registered_models[index];
  }

  ~AMSWrap()
  {
    for (auto E : executors) {
      delete reinterpret_cast<ams::AMSWorkflow *>(E);
    }
    ams::util::close();
  }
};

static std::once_flag _amsInitFlag;
static std::once_flag _amsFinalizeFlag;
static std::unique_ptr<AMSWrap> _amsWrap;

ams::AMSWorkflow *_AMSCreateExecutor(AMSCAbstrModel model,
                                     int process_id,
                                     int world_size)
{
  CFATAL(AMS, _amsWrap == nullptr, "AMSInit has not been called.")
  auto &model_descr = _amsWrap->get_model(model);

  ams::AMSWorkflow *WF = new ams::AMSWorkflow(model_descr.second.SPath,
                                              model_descr.first,
                                              model_descr.second.DBLabel,
                                              model_descr.second.threshold,
                                              model_descr.second.uqPolicy,
                                              process_id,
                                              world_size);
  return WF;
}

AMSExecutor _AMSRegisterExecutor(ams::AMSWorkflow *workflow)
{
  CFATAL(AMS, _amsWrap == nullptr, "AMSInit has not been called.")
  _amsWrap->executors.push_back(static_cast<void *>(workflow));
  return static_cast<AMSExecutor>(_amsWrap->executors.size()) - 1L;
}
}  // namespace

namespace ams
{

void AMSInit()
{
  std::call_once(_amsInitFlag, [&]() {
    DBG(AMS, "Initialization of AMS")
    _amsWrap = std::make_unique<AMSWrap>();
  });
}

void AMSFinalize()
{
  std::call_once(_amsFinalizeFlag, [&]() {
    DBG(AMS, "Finalization of AMS")
    _amsWrap.reset();
  });
}


AMSExecutor AMSCreateExecutor(AMSCAbstrModel model,
                              int process_id,
                              int world_size)
{
  auto *dWF = _AMSCreateExecutor(model, process_id, world_size);
  return _AMSRegisterExecutor(dWF);
}


void AMSExecute(AMSExecutor executor,
                EOSLambda &OrigComputation,
                const ams::SmallVector<ams::AMSTensor> &ins,
                ams::SmallVector<ams::AMSTensor> &inouts,
                ams::SmallVector<ams::AMSTensor> &outs)
{
  int64_t index = static_cast<int64_t>(executor);
  if (index >= _amsWrap->executors.size())
    throw std::runtime_error("AMS Executor identifier does not exist\n");
  auto currExec = _amsWrap->executors[index];

  ams::AMSWorkflow *workflow = reinterpret_cast<ams::AMSWorkflow *>(currExec);
  DBG(AMS,
      "Calling AMS with in:%ld, inout:%ld, out:%ld",
      ins.size(),
      inouts.size(),
      outs.size());

  callAMS(workflow, OrigComputation, ins, inouts, outs);
}

void AMSCExecute(AMSExecutor executor,
                 EOSCFn OrigCComputation,
                 void *args,
                 const ams::SmallVector<ams::AMSTensor> &ins,
                 ams::SmallVector<ams::AMSTensor> &inouts,
                 ams::SmallVector<ams::AMSTensor> &outs)
{

  // Define the lambda and let the compiler deduce the type conversion to std::function
  EOSLambda OrigComputation =
      [&](const ams::SmallVector<ams::AMSTensor> &ams_ins,
          ams::SmallVector<ams::AMSTensor> &ams_inouts,
          ams::SmallVector<ams::AMSTensor> &ams_outs) {
        OrigCComputation(args, ams_ins, ams_inouts, ams_outs);
      };

  AMSExecute(executor, OrigComputation, ins, inouts, outs);
}


void AMSDestroyExecutor(AMSExecutor executor)
{
  CFATAL(AMS, _amsWrap == nullptr, "AMSInit has not been called.")
  int64_t index = static_cast<int64_t>(executor);
  if (index >= _amsWrap->executors.size())
    throw std::runtime_error("AMS Executor identifier does not exist\n");
  auto currExec = _amsWrap->executors[index];

  delete reinterpret_cast<ams::AMSWorkflow *>(currExec);
}


const char *AMSGetAllocatorName(AMSResourceType device)
{
  auto &rm = ams::ResourceManager::getInstance();
  return rm.getAllocatorName(device).c_str();
}

void AMSSetAllocator(AMSResourceType resource, const char *alloc_name)
{
  auto &rm = ams::ResourceManager::getInstance();
  std::string alloc(alloc_name);
  rm.setAllocator(alloc, resource);
}

AMSCAbstrModel AMSRegisterAbstractModel(const char *domain_name,
                                        AMSUQPolicy uq_policy,
                                        double threshold,
                                        const char *surrogate_path,
                                        const char *db_label)
{
  CFATAL(AMS, !_amsWrap, "AMSInit has not been called.")
  auto id = _amsWrap->get_model_index(domain_name);
  if (id == -1) {
    id = _amsWrap->register_model(
        domain_name, uq_policy, threshold, surrogate_path, db_label);
  }

  return id;
}


AMSCAbstrModel AMSQueryModel(const char *domain_model)
{
  CFATAL(AMS, _amsWrap == nullptr, "AMSInit has not been called.")
  return _amsWrap->get_model_index(domain_model);
}

void AMSConfigureFSDatabase(AMSDBType db_type, const char *db_path)
{
  auto &db_instance = ams::db::DBManager::getInstance();
  db_instance.instantiate_fs_db(db_type, std::string(db_path));
}


#ifdef __AMS_ENABLE_MPI__
AMSExecutor AMSCreateDistributedExecutor(AMSCAbstrModel model,
                                         MPI_Comm Comm,
                                         int process_id,
                                         int world_size)

{
  auto *dWF = _AMSCreateExecutor(model, process_id, world_size);
  dWF->set_communicator(Comm);
  return _AMSRegisterExecutor(dWF);
}
#endif
}  // namespace ams
