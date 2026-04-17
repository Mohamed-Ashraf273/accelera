const express = require("express");
const router = express.Router();
const User = require("../schemas/user");
router.get("/", async (req, res) => {
  try {
    const users = await User.find();
    return res.status(200).json(users);
  } catch (err) {
    console.error("Error while fetching Users:", err);
    res.status(500).json({ message: "There is an error while fetching Users" });
  }
});
router.post("/", async (req, res) => {
  try {
    let { name, email } = req.body;
    const user = new User({
      name,
      email,
    });
    await user.save();
    res.status(201).json(user);
  } catch (err) {
    console.error("Error while creating Users:", err);
    res.status(500).json({ message: "There is an error while creating Users" });
  }
});
module.exports = router;
