const mongoose = require("mongoose");
const userSChema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
  },
  email: {
    type: String,
    required: true,
    unique: true,
  },
  role: {
    type: String,
    required: true,
    enum: ["admin", "user"],
  },
});
const User = mongoose.model("User", userSChema);
module.exports = User;
