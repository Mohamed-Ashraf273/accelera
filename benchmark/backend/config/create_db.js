const mongoose = require("mongoose")

const connectMongo = async (Mongos_url) => {
  try {
    const connection = await mongoose.connect(Mongos_url);
    console.log("The Connection to MongoDB is created successfully");
  } catch (error) {
    console.log("Error when trying to connect to MongoDB:", error);
  }
};
module.exports = connectMongo
