const mongoose = require("mongoose")
const dotenv = require("dotenv")
dotenv.config()
const connectMongo = async () => {
    try {
    const connection = await mongoose.connect(process.env.Mongos_url)
    console.log("The Connection to MongoDB is created successfully")
    }
    catch (error) {
        console.log("Error when trying to connect to MongoDB:", error)
    }
}
module.exports = connectMongo