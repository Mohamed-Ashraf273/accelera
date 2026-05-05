const express = require("express");
const router = express.Router();
const User = require("../schemas/user");

router.post("/login", async (req, res) => {
  try {
    const {  email } = req.body;
    const user = await User.findOne({ email: email });
    if (!user) {
      return res
        .status(404)
        .json({ message: `There is no user with these info` });
    }
    return res.status(200).json(user);
  } catch (err) {
    console.error("Error while fetching Users:", err);
    res.status(500).json({ message: "There is an error while fetching Users" });
  }
});
router.get("/", async (req, res) => {
  try {
    const users = await User.find();
    return res.status(200).json(users);
  } catch (err) {
    console.error("Error while fetching Users:", err);
    res.status(500).json({ message: "There is an error while fetching Users" });
  }
});

router.post("/signup", async (req, res) => {
  try {
    let { name, email, role } = req.body;
    role = role.toLowerCase();
    const user = new User({
      name,
      email,
      role,
    });
    await user.save();
    res.status(201).json(user);
  } catch (err) {
    console.error("Error while creating Users:", err);
    res.status(500).json({ message: "There is an error while creating Users" });
  }
});
module.exports = router;
