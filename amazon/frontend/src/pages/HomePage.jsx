import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import {
  getOutgoingFriendReqs,
  getRecommendedUsers,
  getUserFriends,
  sendFriendRequest,
} from "../lib/api";
import { Link } from "react-router";
import { CheckCircleIcon, MapPinIcon, UserPlusIcon, UsersIcon } from "lucide-react";

import FriendCard from "../components/FriendCard";
import NoFriendsFound from "../components/NoFriendsFound";

const QUERY_KEYS = {
  FRIENDS: ["friends"],
  USERS: ["users"],
  OUTGOING_REQUESTS: ["outgoingFriendReqs"],
};

const LoadingSpinner = () => (
  <div className="flex justify-center py-12">
    <span className="loading loading-spinner loading-lg" />
  </div>
);

const EmptyRecommendations = () => (
  <div className="card bg-base-200 p-6 text-center">
    <h3 className="font-semibold text-lg mb-2">No recommendations available</h3>
    <p className="text-base-content opacity-70">
      Check back later for new language partners!
    </p>
  </div>
);

const UserAvatar = ({ imageUrl, name }) => (
  <div className="avatar size-16 rounded-full">
    <img src={imageUrl} alt={name} />
  </div>
);

const UserLocation = ({ location }) => {
  if (!location) return null;

  return (
    <div className="flex items-center text-xs opacity-70 mt-1">
      <MapPinIcon className="size-3 mr-1" />
      {location}
    </div>
  );
};

const RequestButton = ({ isSent, isProcessing, onClick }) => {
  const buttonClass = isSent ? "btn-disabled" : "btn-primary";
  const isDisabled = isSent || isProcessing;

  return (
    <button
      className={`btn w-full mt-2 ${buttonClass}`}
      onClick={onClick}
      disabled={isDisabled}
    >
      {isSent ? (
        <>
          <CheckCircleIcon className="size-4 mr-2" />
          Request Sent
        </>
      ) : (
        <>
          <UserPlusIcon className="size-4 mr-2" />
          Send Friend Request
        </>
      )}
    </button>
  );
};

const UserCard = ({ user, onSendRequest, isPending, sentRequestIds }) => {
  const requestAlreadySent = sentRequestIds.has(user._id);

  return (
    <div className="card bg-base-200 hover:shadow-lg transition-all duration-300">
      <div className="card-body p-5 space-y-4">
        <div className="flex items-center gap-3">
          <UserAvatar imageUrl={user.profilePic} name={user.fullName} />
          <div>
            <h3 className="font-semibold text-lg">{user.fullName}</h3>
            <UserLocation location={user.location} />
          </div>
        </div>

        {user.bio && <p className="text-sm opacity-70">{user.bio}</p>}

        <RequestButton
          isSent={requestAlreadySent}
          isProcessing={isPending}
          onClick={() => onSendRequest(user._id)}
        />
      </div>
    </div>
  );
};

const HomePage = () => {
  const queryClient = useQueryClient();
  const [sentRequestsSet, setSentRequestsSet] = useState(new Set());

  const friendsQuery = useQuery({
    queryKey: QUERY_KEYS.FRIENDS,
    queryFn: getUserFriends,
  });

  const usersQuery = useQuery({
    queryKey: QUERY_KEYS.USERS,
    queryFn: getRecommendedUsers,
  });

  const outgoingRequestsQuery = useQuery({
    queryKey: QUERY_KEYS.OUTGOING_REQUESTS,
    queryFn: getOutgoingFriendReqs,
  });

  const requestMutation = useMutation({
    mutationFn: sendFriendRequest,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.OUTGOING_REQUESTS });
    },
  });

  const friendsList = friendsQuery.data ?? [];
  const usersList = usersQuery.data ?? [];

  useEffect(() => {
    if (!outgoingRequestsQuery.data?.length) return;

    const requestIds = new Set(
      outgoingRequestsQuery.data.map(req => req.recipient._id)
    );

    setSentRequestsSet(requestIds);
  }, [outgoingRequestsQuery.data]);

  const handleSendRequest = (userId) => {
    requestMutation.mutate(userId);
  };

  const showFriendsLoading = friendsQuery.isLoading;
  const showUsersLoading = usersQuery.isLoading;
  const hasFriends = friendsList.length > 0;
  const hasUsers = usersList.length > 0;

  return (
    <div className="p-4 sm:p-6 lg:p-8">
      <div className="container mx-auto space-y-10">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight">Your Friends</h2>
          <Link to="/notifications" className="btn btn-outline btn-sm">
            <UsersIcon className="mr-2 size-4" />
            Friend Requests
          </Link>
        </div>

        {showFriendsLoading ? (
          <LoadingSpinner />
        ) : !hasFriends ? (
          <NoFriendsFound />
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {friendsList.map((friend) => (
              <FriendCard key={friend._id} friend={friend} />
            ))}
          </div>
        )}

        <section>
          <div className="mb-6 sm:mb-8">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div>
                <h2 className="text-2xl sm:text-3xl font-bold tracking-tight">Meet New Learners</h2>
                <p className="opacity-70">
                  Discover perfect language exchange partners based on your profile
                </p>
              </div>
            </div>
          </div>

          {showUsersLoading ? (
            <LoadingSpinner />
          ) : !hasUsers ? (
            <EmptyRecommendations />
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {usersList.map((user) => (
                <UserCard
                  key={user._id}
                  user={user}
                  onSendRequest={handleSendRequest}
                  isPending={requestMutation.isPending}
                  sentRequestIds={sentRequestsSet}
                />
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
};

export default HomePage;