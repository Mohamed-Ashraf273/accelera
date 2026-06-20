const express = require("express");
const router = express.Router();
const User = require("../schemas/user");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const auth = require("../middleware/auth");
const admin = require("../middleware/admin");

const createToken = (user) => {
  return jwt.sign(
    { id: user._id, role: user.role },
    process.env.JWT_SECRET || "accelera_secret",
    { expiresIn: "7d" },
  );
};

router.post("/login", async (req, res) => {
  try {
    let { email, password } = req.body;
    if (!email || !password) {
      return res.status(400).json({ message: "email and password must exist" });
    }
    email = email.toLowerCase();
    const user = await User.findOne({ email: email });
    if (!user) {
      return res
        .status(404)
        .json({ message: `There is no user with these info` });
    }
    if (!user.password) {
      return res.status(400).json({
        message: "This account has no password",
      });
    }
    const isCorrectPassword = await bcrypt.compare(password, user.password);
    if (!isCorrectPassword) {
      return res.status(400).json({ message: "Wrong password" });
    }
    const token = createToken(user);
    const userObj = user.toObject();
    delete userObj.password;
    return res.status(200).json({ user: userObj, token });
  } catch (err) {
    console.error("Error while fetching Users:", err);
    res.status(500).json({ message: "There is an error while fetching Users" });
  }
});
router.get("/", auth, admin, async (req, res) => {
  try {
    const users = await User.find().select("-password");
    return res.status(200).json(users);
  } catch (err) {
    console.error("Error while fetching Users:", err);
    res.status(500).json({ message: "There is an error while fetching Users" });
  }
});

router.post("/signup", async (req, res) => {
  try {
    let { name, email, password } = req.body;
    if (!name || !email || !password) {
      return res
        .status(400)
        .json({ message: "name email and password must exist" });
    }
    name = name.toLowerCase();
    email = email.toLowerCase();
    const role = email === process.env.ADMIN_EMAIL ? "admin" : "user";
    const userByName = await User.findOne({ name });
    if (userByName)
      return res.status(400).json({
        message: `There is name  ${name} already exist`,
      });
    const userByEmail = await User.findOne({ email });
    if (userByEmail)
      return res.status(400).json({
        message: `There is email  ${email} already exist`,
      });
    const hashedPassword = await bcrypt.hash(password, 10);
    const user = new User({
      name,
      email,
      password: hashedPassword,
      role,
    });
    await user.save();
    const userObj = user.toObject();
    delete userObj.password;
    res.status(201).json(userObj);
  } catch (err) {
    console.error("Error while creating Users:", err);
    res.status(500).json({ message: "There is an error while creating Users" });
  }
});

router.delete("/:id", auth, admin, async (req, res) => {
  try {
    const userId = req.params.id;
    if (req.user._id.toString() === userId) {
      return res.status(400).json({ message: "Admin cannot delete himself" });
    }
    const user = await User.findByIdAndDelete(userId);
    if (!user) {
      return res
        .status(404)
        .json({ message: `There is no user for this id: ${userId}` });
    }
    return res.status(200).json({ message: "user successfully deleted" });
  } catch (err) {
    console.error("Error while deleting Users:", err);
    res.status(500).json({ message: "There is an error while deleting Users" });
  }
});
module.exports = router;
