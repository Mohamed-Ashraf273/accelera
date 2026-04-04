const mongoose = require('mongoose')
const userSChema = new mongoose.Schema({
    name: {
        type: String,
        required: true
    }
});
const User = mongoose.model('User', userSChema)
module.exports = User
