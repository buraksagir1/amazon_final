import { upsertStreamUser } from "../lib/stream.js";
import User from "../models/User.js";
import jwt from "jsonwebtoken";

const generateAuthToken = (userId) => {
  return jwt.sign({ userId }, process.env.JWT_SECRET_KEY, { expiresIn: "7d" });
};

const setAuthCookie = (res, token) => {
  const WEEK_IN_MS = 604800000;
  res.cookie("jwt", token, {
    maxAge: WEEK_IN_MS,
    httpOnly: true,
    sameSite: "strict",
    secure: process.env.NODE_ENV === "production",
  });
};

const validateEmail = (email) => {
  const pattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return pattern.test(email);
};

const getRandomAvatarUrl = () => {
  const randomIndex = Math.floor(Math.random() * 100) + 1;
  return `https://avatar.iran.liara.run/public/${randomIndex}.png`;
};

export async function signup(req, res) {
  const { email, password, fullName } = req.body;

  try {
    const missingFields = [];
    if (!email) missingFields.push("email");
    if (!password) missingFields.push("password");
    if (!fullName) missingFields.push("fullName");

    if (missingFields.length > 0) {
      return res.status(400).json({
        message: "Please provide all required information",
        fields: missingFields
      });
    }

    if (password.length < 6) {
      return res.status(400).json({
        message: "Your password needs to be at least 6 characters"
      });
    }

    if (!validateEmail(email)) {
      return res.status(400).json({
        message: "Please provide a valid email address"
      });
    }

    const existingAccount = await User.findOne({ email });
    if (existingAccount) {
      return res.status(400).json({
        message: "An account with this email already exists"
      });
    }

    const profilePicture = getRandomAvatarUrl();

    const userAccount = await User.create({
      email,
      fullName,
      password,
      profilePic: profilePicture,
    });

    const token = generateAuthToken(userAccount._id);
    setAuthCookie(res, token);

    try {
      await upsertStreamUser({
        id: userAccount._id.toString(),
        name: userAccount.fullName,
        image: userAccount.profilePic || "",
      });
      console.log(`Stream account created: ${userAccount.fullName}`);
    } catch (streamErr) {
      console.log("Stream user creation failed:", streamErr);
    }

    return res.status(201).json({ success: true, user: userAccount });
  } catch (error) {
    console.log("Signup failed:", error);
    return res.status(500).json({ message: "Something went wrong on our end" });
  }
}

export async function login(req, res) {
  const { email, password } = req.body;

  try {
    if (!email || !password) {
      return res.status(400).json({
        message: "Both email and password are required"
      });
    }

    const userAccount = await User.findOne({ email });

    if (!userAccount) {
      return res.status(401).json({
        message: "The email or password you entered is incorrect"
      });
    }

    const isValidPassword = await userAccount.matchPassword(password);

    if (!isValidPassword) {
      return res.status(401).json({
        message: "The email or password you entered is incorrect"
      });
    }

    const token = generateAuthToken(userAccount._id);
    setAuthCookie(res, token);

    return res.status(200).json({ success: true, user: userAccount });
  } catch (error) {
    console.log("Login failed:", error.message);
    return res.status(500).json({ message: "Something went wrong on our end" });
  }
}

export function logout(req, res) {
  res.clearCookie("jwt");
  return res.status(200).json({
    success: true,
    message: "You have been logged out successfully"
  });
}

export async function onboard(req, res) {
  const userId = req.user._id;
  const { fullName, bio, location } = req.body;

  try {
    const requiredFields = { fullName, bio, location };
    const emptyFields = Object.keys(requiredFields).filter(key => !requiredFields[key]);

    if (emptyFields.length > 0) {
      return res.status(400).json({
        message: "Please fill in all the required fields",
        missingFields: emptyFields,
      });
    }

    const profileData = {
      ...req.body,
      isOnboarded: true,
    };

    const updatedProfile = await User.findByIdAndUpdate(userId, profileData, {
      new: true
    });

    if (!updatedProfile) {
      return res.status(404).json({ message: "Could not find your account" });
    }

    try {
      await upsertStreamUser({
        id: updatedProfile._id.toString(),
        name: updatedProfile.fullName,
        image: updatedProfile.profilePic || "",
      });
      console.log(`Stream profile updated: ${updatedProfile.fullName}`);
    } catch (streamErr) {
      console.log("Stream update failed:", streamErr.message);
    }

    return res.status(200).json({ success: true, user: updatedProfile });
  } catch (error) {
    console.error("Onboarding failed:", error);
    return res.status(500).json({ message: "Something went wrong on our end" });
  }
}