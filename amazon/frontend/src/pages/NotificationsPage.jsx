import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { acceptFriendRequest, getFriendRequests } from "../lib/api";
import { BellIcon, ClockIcon, MessageSquareIcon, UserCheckIcon } from "lucide-react";
import NoNotificationsFound from "../components/NoNotificationsFound";

const QUERY_KEYS = {
  FRIEND_REQUESTS: ["friendRequests"],
  FRIENDS: ["friends"],
};

const BADGE_STYLES = {
  PRIMARY: "badge badge-primary badge-sm",
  OUTLINE: "badge badge-outline badge-sm",
  SUCCESS: "badge badge-success",
};

const LoadingView = () => (
  <div className="flex justify-center py-12">
    <span className="loading loading-spinner loading-lg"></span>
  </div>
);

const SectionHeader = ({ icon: Icon, title, count, iconColor }) => (
  <h2 className="text-xl font-semibold flex items-center gap-2">
    <Icon className={`h-5 w-5 ${iconColor}`} />
    {title}
    {count > 0 && <span className="badge badge-primary ml-2">{count}</span>}
  </h2>
);

const UserAvatar = ({ imageUrl, name, size = "w-14 h-14" }) => (
  <div className={`avatar ${size} rounded-full bg-base-300`}>
    <img src={imageUrl} alt={name} />
  </div>
);

const LanguageBadges = ({ native, learning }) => (
  <div className="flex flex-wrap gap-1.5 mt-1">
    <span className={BADGE_STYLES.PRIMARY}>Native: {native}</span>
    <span className={BADGE_STYLES.OUTLINE}>Learning: {learning}</span>
  </div>
);

const AcceptButton = ({ onClick, isDisabled }) => (
  <button
    className="btn btn-primary btn-sm"
    onClick={onClick}
    disabled={isDisabled}
  >
    Accept
  </button>
);

const RequestCard = ({ request, onAccept, isProcessing }) => {
  const { sender } = request;

  return (
    <div className="card bg-base-200 shadow-sm hover:shadow-md transition-shadow">
      <div className="card-body p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <UserAvatar imageUrl={sender.profilePic} name={sender.fullName} />
            <div>
              <h3 className="font-semibold">{sender.fullName}</h3>
              <LanguageBadges
                native={sender.nativeLanguage}
                learning={sender.learningLanguage}
              />
            </div>
          </div>

          <AcceptButton
            onClick={() => onAccept(request._id)}
            isDisabled={isProcessing}
          />
        </div>
      </div>
    </div>
  );
};

const ConnectionNotification = ({ notification }) => {
  const { recipient } = notification;

  return (
    <div className="card bg-base-200 shadow-sm">
      <div className="card-body p-4">
        <div className="flex items-start gap-3">
          <UserAvatar
            imageUrl={recipient.profilePic}
            name={recipient.fullName}
            size="mt-1 size-10"
          />
          <div className="flex-1">
            <h3 className="font-semibold">{recipient.fullName}</h3>
            <p className="text-sm my-1">
              {recipient.fullName} accepted your friend request
            </p>
            <p className="text-xs flex items-center opacity-70">
              <ClockIcon className="h-3 w-3 mr-1" />
              Recently
            </p>
          </div>
          <div className={BADGE_STYLES.SUCCESS}>
            <MessageSquareIcon className="h-3 w-3 mr-1" />
            New Friend
          </div>
        </div>
      </div>
    </div>
  );
};

const IncomingRequestsSection = ({ requests, onAccept, isProcessing }) => {
  if (requests.length === 0) return null;

  return (
    <section className="space-y-4">
      <SectionHeader
        icon={UserCheckIcon}
        title="Friend Requests"
        count={requests.length}
        iconColor="text-primary"
      />
      <div className="space-y-3">
        {requests.map((request) => (
          <RequestCard
            key={request._id}
            request={request}
            onAccept={onAccept}
            isProcessing={isProcessing}
          />
        ))}
      </div>
    </section>
  );
};

const AcceptedRequestsSection = ({ notifications }) => {
  if (notifications.length === 0) return null;

  return (
    <section className="space-y-4">
      <SectionHeader
        icon={BellIcon}
        title="New Connections"
        iconColor="text-success"
      />
      <div className="space-y-3">
        {notifications.map((notification) => (
          <ConnectionNotification key={notification._id} notification={notification} />
        ))}
      </div>
    </section>
  );
};

const NotificationsPage = () => {
  const queryClient = useQueryClient();

  const requestsQuery = useQuery({
    queryKey: QUERY_KEYS.FRIEND_REQUESTS,
    queryFn: getFriendRequests,
  });

  const acceptMutation = useMutation({
    mutationFn: acceptFriendRequest,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.FRIEND_REQUESTS });
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.FRIENDS });
    },
  });

  const handleAcceptRequest = (requestId) => {
    acceptMutation.mutate(requestId);
  };

  const pendingRequests = requestsQuery.data?.incomingReqs ?? [];
  const acceptedConnections = requestsQuery.data?.acceptedReqs ?? [];

  const hasNoNotifications = pendingRequests.length === 0 && acceptedConnections.length === 0;

  return (
    <div className="p-4 sm:p-6 lg:p-8">
      <div className="container mx-auto max-w-4xl space-y-8">
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight mb-6">Notifications</h1>

        {requestsQuery.isLoading ? (
          <LoadingView />
        ) : (
          <>
            <IncomingRequestsSection
              requests={pendingRequests}
              onAccept={handleAcceptRequest}
              isProcessing={acceptMutation.isPending}
            />

            <AcceptedRequestsSection notifications={acceptedConnections} />

            {hasNoNotifications && <NoNotificationsFound />}
          </>
        )}
      </div>
    </div>
  );
};

export default NotificationsPage;