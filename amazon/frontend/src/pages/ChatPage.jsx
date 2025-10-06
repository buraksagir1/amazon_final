import { useEffect, useState } from "react";
import { useParams } from "react-router";
import useAuthUser from "../hooks/useAuthUser";
import { useQuery } from "@tanstack/react-query";
import { getStreamToken } from "../lib/api";

import {
  Channel,
  ChannelHeader,
  Chat,
  MessageInput,
  MessageList,
  Thread,
  Window,
} from "stream-chat-react";
import { StreamChat } from "stream-chat";
import toast from "react-hot-toast";

import ChatLoader from "../components/ChatLoader";
import CallButton from "../components/CallButton";

const STREAM_API_KEY = import.meta.env.VITE_STREAM_API_KEY;

const ChatPage = () => {
  const { id: recipientUserId } = useParams();

  const [streamClient, setStreamClient] = useState(null);
  const [activeChannel, setActiveChannel] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  const { authUser } = useAuthUser();

  const { data: streamTokenData } = useQuery({
    queryKey: ["streamToken"],
    queryFn: getStreamToken,
    enabled: !!authUser,
  });

  useEffect(() => {
    const setupChatConnection = async () => {
      if (!streamTokenData?.token || !authUser) return;

      try {
        console.log("Setting up stream chat connection...");

        const chatInstance = StreamChat.getInstance(STREAM_API_KEY);

        await chatInstance.connectUser(
          {
            id: authUser._id,
            name: authUser.fullName,
            image: authUser.profilePic,
          },
          streamTokenData.token
        );

        const conversationId = [authUser._id, recipientUserId].sort().join("-");

        const chatChannel = chatInstance.channel("messaging", conversationId, {
          members: [authUser._id, recipientUserId],
        });

        await chatChannel.watch();

        setStreamClient(chatInstance);
        setActiveChannel(chatChannel);
      } catch (error) {
        console.error("Failed to establish chat connection:", error);
        toast.error("Unable to connect to chat. Please refresh and try again.");
      } finally {
        setIsLoading(false);
      }
    };

    setupChatConnection();
  }, [streamTokenData, authUser, recipientUserId]);

  const initiateVideoCall = () => {
    if (activeChannel) {
      const videoCallUrl = `${window.location.origin}/call/${activeChannel.id}`;

      activeChannel.sendMessage({
        text: `I've started a video call. Join me here: ${videoCallUrl}`,
      });

      toast.success("Video call invitation sent!");
    }
  };

  if (isLoading || !streamClient || !activeChannel) return <ChatLoader />;

  return (
    <div className="h-[93vh]">
      <Chat client={streamClient}>
        <Channel channel={activeChannel}>
          <div className="w-full relative">
            <CallButton handleVideoCall={initiateVideoCall} />
            <Window>
              <ChannelHeader />
              <MessageList />
              <MessageInput focus />
            </Window>
          </div>
          <Thread />
        </Channel>
      </Chat>
    </div>
  );
};
export default ChatPage;