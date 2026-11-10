#include <unistd.h>

#define CATCH_CONFIG_PREFIX_ALL
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "AMS.h"
#include "wf/basedb.hpp"

namespace fs = std::filesystem;

CATCH_TEST_CASE("Filesystem DB configuration recovers from invalid paths",
                "[ams][db][configuration]")
{
  const auto root = fs::temp_directory_path() /
                    ("ams_db_recovery_" + std::to_string(getpid()));
  fs::create_directories(root);
  const auto valid_path = root / "valid";
  const auto missing_path = root / "missing";
  const auto file_path = root / "regular_file";
  fs::create_directory(valid_path);
  std::ofstream(file_path).put('x');

  auto& db = ams::db::DBManager::getInstance();
  CATCH_REQUIRE_FALSE(db.isInitialized());
  CATCH_REQUIRE_THROWS_AS(ams::AMSConfigureFSDatabase(ams::AMSDBType::AMS_HDF5,
                                                      missing_path.c_str()),
                          std::runtime_error);
  CATCH_CHECK_FALSE(db.isInitialized());
  std::string unconfigured_domain = "unconfigured_domain";
  CATCH_CHECK(db.getDB(unconfigured_domain) == nullptr);

  CATCH_REQUIRE_THROWS_AS(ams::AMSConfigureFSDatabase(ams::AMSDBType::AMS_HDF5,
                                                      file_path.c_str()),
                          std::runtime_error);
  CATCH_CHECK_FALSE(db.isInitialized());

  CATCH_REQUIRE_NOTHROW(ams::AMSConfigureFSDatabase(ams::AMSDBType::AMS_HDF5,
                                                    valid_path.c_str()));
  CATCH_REQUIRE(db.isInitialized());

  CATCH_REQUIRE_THROWS_AS(ams::AMSConfigureFSDatabase(ams::AMSDBType::AMS_JSON,
                                                      missing_path.c_str()),
                          std::runtime_error);
  CATCH_REQUIRE_THROWS_AS(ams::AMSConfigureFSDatabase(ams::AMSDBType::AMS_JSON,
                                                      file_path.c_str()),
                          std::runtime_error);

  std::string retained_domain = "retained_domain";
  auto retained_db = db.getDB(retained_domain);
  CATCH_REQUIRE(retained_db != nullptr);
  CATCH_CHECK(retained_db->dbType() == ams::AMSDBType::AMS_HDF5);
  CATCH_CHECK(fs::path(retained_db->getFilename()).parent_path() == valid_path);
  retained_db.reset();
  db.clean();

  fs::remove(valid_path / "retained_domain_0.h5");
  fs::remove(valid_path);
  std::string disappeared_domain = "disappeared_domain";
  CATCH_REQUIRE_THROWS_AS(db.getDB(disappeared_domain), std::runtime_error);
  CATCH_REQUIRE_THROWS_AS(ams::db::hdf5DB(valid_path.string(),
                                          "direct_domain",
                                          0),
                          std::runtime_error);
  CATCH_REQUIRE_THROWS_AS(ams::db::hdf5DB(file_path.string(),
                                          "direct_domain",
                                          0),
                          std::runtime_error);

  fs::create_directory(valid_path);
  auto recovered_db = db.getDB(disappeared_domain);
  CATCH_REQUIRE(recovered_db != nullptr);
  CATCH_CHECK(fs::exists(recovered_db->getFilename()));
  recovered_db.reset();
  db.clean();
  fs::remove_all(root);
}
