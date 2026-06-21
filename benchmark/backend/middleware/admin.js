const admin = (req, res, next) => {
  if (!req.user || req.user.role !== "admin") {
    return res.status(403).json({ message: "You ar enot an admin" });
  }
  next();
};

module.exports = admin;
