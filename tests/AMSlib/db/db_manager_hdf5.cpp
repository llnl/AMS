#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "wf/basedb.hpp"

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::cout << "Wrong command line argument :\n";
    std::cout << argv[0] << " <path-for-db-manager>\n";
    return -1;
  }
  std::string db_path(argv[1]);
  auto& db_instance = ams::db::DBManager::getInstance();
  db_instance.instantiate_fs_db(AMSDBType::AMS_HDF5, db_path);
  for (auto dn : {std::string("domain_1"),
                  std::string("domain_2"),
                  std::string("domain_1"),
                  std::string("domain_2")}) {
    auto file_db = db_instance.getDB(dn, dn);
  }

  if (db_instance.getNumInstances() != 2) {
    std::cout << "Wrong number of instances\n";
    return -1;
  }

  // This is done to internally call the de-constructors of the respective DB.
  db_instance.clean();
  if (db_instance.getNumInstances() != 0) {
    std::cout << "DB Instances did not reset \n";
    return -1;
  }

  for (auto dn : {std::string("domain_1"), std::string("domain_2")}) {
    auto fn = db_path + dn + "_0.h5";

    if (!std::filesystem::exists(fn)) {
      std::cout << "File " << fn << "does not exists.\n";
      return -1;
    }
  }

  return 0;
}
