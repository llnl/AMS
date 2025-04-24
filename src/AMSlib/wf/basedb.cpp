#include <string>

#include "AMS.h"

namespace ams
{
namespace db
{
/**
   * @brief get the type of the database from a string
   * @param[in] type the type of the db in string format
   * @retrun the database type
   */
AMSDBType getDBType(std::string type)
{
  if (type.compare("hdf5") == 0) {
    return AMSDBType::AMS_HDF5;
  } else if (type.compare("rmq") == 0) {
    return AMSDBType::AMS_RMQ;
  }
  return AMSDBType::AMS_NONE;
}

std::string getDBTypeAsStr(AMSDBType type)
{
  switch (type) {
    case AMSDBType::AMS_NONE:
      return "None";
    case AMSDBType::AMS_HDF5:
      return "hdf5";
    case AMSDBType::AMS_RMQ:
      return "rmq";
  }
  return "Unknown";
}


}  // namespace db
}  // namespace ams
