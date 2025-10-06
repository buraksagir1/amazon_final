import User from "../models/User.js";
import FriendRequest from "../models/FriendRequest.js";

const FRIEND_REQUEST_STATUS = {
  PENDING: "pending",
  ACCEPTED: "accepted",
};

const USER_POPULATE_FIELDS = "fullName profilePic nativeLanguage learningLanguage";
const BASIC_USER_FIELDS = "fullName profilePic";

const buildUserQuery = (userId, friendsList) => ({
  $and: [
    { _id: { $ne: userId } },
    { _id: { $nin: friendsList } },
    { isOnboarded: true },
  ],
});

const checkIfUsersAreFriends = (friendsList, userId) => {
  return friendsList.includes(userId);
};

const findExistingRequestBetweenUsers = async (user1, user2) => {
  return await FriendRequest.findOne({
    $or: [
      { sender: user1, recipient: user2 },
      { sender: user2, recipient: user1 },
    ],
  });
};

const updateBothUsersFriendsList = async (userId1, userId2) => {
  await Promise.all([
    User.findByIdAndUpdate(userId1, {
      $addToSet: { friends: userId2 },
    }),
    User.findByIdAndUpdate(userId2, {
      $addToSet: { friends: userId1 },
    }),
  ]);
};

export async function getRecommendedUsers(req, res) {
  try {
    const { id: userId, friends: friendsList } = req.user;

    const query = buildUserQuery(userId, friendsList);
    const users = await User.find(query);

    return res.status(200).json(users);
  } catch (error) {
    console.error("Recommendation fetch error:", error.message);
    return res.status(500).json({ message: "Failed to load user recommendations" });
  }
}

export async function getMyFriends(req, res) {
  try {
    const userWithFriends = await User.findById(req.user.id)
      .select("friends")
      .populate("friends", USER_POPULATE_FIELDS);

    const friendsList = userWithFriends?.friends || [];

    return res.status(200).json(friendsList);
  } catch (error) {
    console.error("Friends list error:", error.message);
    return res.status(500).json({ message: "Unable to fetch your friends list" });
  }
}

export async function sendFriendRequest(req, res) {
  const currentUserId = req.user.id;
  const { id: targetUserId } = req.params;

  try {
    if (currentUserId === targetUserId) {
      return res.status(400).json({ message: "You cannot add yourself as a friend" });
    }

    const recipientUser = await User.findById(targetUserId);

    if (!recipientUser) {
      return res.status(404).json({ message: "The user you're looking for doesn't exist" });
    }

    const alreadyFriends = checkIfUsersAreFriends(recipientUser.friends, currentUserId);

    if (alreadyFriends) {
      return res.status(400).json({ message: "You're already friends with this user" });
    }

    const existingRequest = await findExistingRequestBetweenUsers(currentUserId, targetUserId);

    if (existingRequest) {
      return res.status(400).json({ message: "A friend request between you already exists" });
    }

    const createdRequest = await FriendRequest.create({
      sender: currentUserId,
      recipient: targetUserId,
    });

    return res.status(201).json(createdRequest);
  } catch (error) {
    console.error("Friend request error:", error.message);
    return res.status(500).json({ message: "Failed to process your friend request" });
  }
}

export async function acceptFriendRequest(req, res) {
  const { id: requestId } = req.params;
  const currentUserId = req.user.id;

  try {
    const friendRequest = await FriendRequest.findById(requestId);

    if (!friendRequest) {
      return res.status(404).json({ message: "This friend request no longer exists" });
    }

    const isRecipient = friendRequest.recipient.toString() === currentUserId;

    if (!isRecipient) {
      return res.status(403).json({ message: "You don't have permission to accept this request" });
    }

    friendRequest.status = FRIEND_REQUEST_STATUS.ACCEPTED;
    await friendRequest.save();

    await updateBothUsersFriendsList(friendRequest.sender, friendRequest.recipient);

    return res.status(200).json({ message: "Friend request accepted successfully" });
  } catch (error) {
    console.log("Accept request error:", error.message);
    return res.status(500).json({ message: "Unable to accept this friend request" });
  }
}

export async function getFriendRequests(req, res) {
  const userId = req.user.id;

  try {
    const [incomingRequests, acceptedRequests] = await Promise.all([
      FriendRequest.find({
        recipient: userId,
        status: FRIEND_REQUEST_STATUS.PENDING,
      })
        .populate("sender", USER_POPULATE_FIELDS)
        .lean(),

      FriendRequest.find({
        sender: userId,
        status: FRIEND_REQUEST_STATUS.ACCEPTED,
      })
        .populate("recipient", BASIC_USER_FIELDS)
        .lean(),
    ]);

    // Populate edilemeyen (sender veya recipient null olan) kayıtları filtrele
    const validIncomingRequests = incomingRequests.filter(req => req.sender !== null);
    const validAcceptedRequests = acceptedRequests.filter(req => req.recipient !== null);

    // Geçersiz kayıtları logla
    const invalidIncoming = incomingRequests.filter(req => req.sender === null);
    const invalidAccepted = acceptedRequests.filter(req => req.recipient === null);

    if (invalidIncoming.length > 0) {
      console.warn(`Found ${invalidIncoming.length} incoming requests with null sender`);
    }

    if (invalidAccepted.length > 0) {
      console.warn(`Found ${invalidAccepted.length} accepted requests with null recipient`);
    }

    return res.status(200).json({
      incomingReqs: validIncomingRequests,
      acceptedReqs: validAcceptedRequests,
    });
  } catch (error) {
    console.log("Request fetch error:", error.message);
    return res.status(500).json({ message: "Failed to retrieve friend requests" });
  }
}

export async function getOutgoingFriendReqs(req, res) {
  const userId = req.user.id;

  try {
    const outgoingRequests = await FriendRequest.find({
      sender: userId,
      status: FRIEND_REQUEST_STATUS.PENDING,
    })
      .populate("recipient", USER_POPULATE_FIELDS)
      .lean();

    // Null recipient kontrolü
    const validRequests = outgoingRequests.filter(req => req.recipient !== null);

    const invalidRequests = outgoingRequests.filter(req => req.recipient === null);
    if (invalidRequests.length > 0) {
      console.warn(`Found ${invalidRequests.length} outgoing requests with null recipient`);
    }

    return res.status(200).json(validRequests);
  } catch (error) {
    console.log("Outgoing requests error:", error.message);
    return res.status(500).json({ message: "Unable to load your sent requests" });
  }
}

export async function getUserStatus(req, res) {
  const { id: targetUserId } = req.params;

  try {
    const user = await User.findById(targetUserId).select("fullName isOnboarded");

    if (!user) {
      return res.status(404).json({ message: "This user doesn't exist" });
    }

    const userStatus = {
      exists: true,
      fullName: user.fullName,
      isOnboarded: user.isOnboarded,
    };

    return res.status(200).json(userStatus);
  } catch (error) {
    console.error("User status error:", error.message);
    return res.status(500).json({ message: "Could not verify user information" });
  }
}
