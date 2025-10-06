import { useState } from "react";
import useAuthUser from "../hooks/useAuthUser";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import toast from "react-hot-toast";
import { completeOnboarding } from "../lib/api";
import { LoaderIcon, MapPinIcon, ShipWheelIcon, ShuffleIcon } from "lucide-react";

const AVATAR_API_BASE = "https://avatar.iran.liara.run/public";
const AVATAR_RANGE = { MIN: 1, MAX: 100 };

const TOAST_MESSAGES = {
  SUCCESS: "Profile onboarded successfully",
  AVATAR_GENERATED: "Random profile picture generated!",
};

const generateRandomAvatarUrl = () => {
  const randomIndex = Math.floor(Math.random() * AVATAR_RANGE.MAX) + AVATAR_RANGE.MIN;
  return `${AVATAR_API_BASE}/${randomIndex}.png`;
};

const ProfilePicturePlaceholder = () => (
  <div className="flex items-center justify-center h-full">
    <CameraIcon className="size-12 text-base-content opacity-40" />
  </div>
);

const ProfilePicturePreview = ({ imageUrl }) => {
  if (!imageUrl) return <ProfilePicturePlaceholder />;

  return (
    <img
      src={imageUrl}
      alt="Profile Preview"
      className="w-full h-full object-cover"
    />
  );
};

const AvatarSection = ({ currentAvatar, onGenerateRandom }) => (
  <div className="flex flex-col items-center justify-center space-y-4">
    <div className="size-32 rounded-full bg-base-300 overflow-hidden">
      <ProfilePicturePreview imageUrl={currentAvatar} />
    </div>

    <div className="flex items-center gap-2">
      <button type="button" onClick={onGenerateRandom} className="btn btn-accent">
        <ShuffleIcon className="size-4 mr-2" />
        Generate Random Avatar
      </button>
    </div>
  </div>
);

const FormField = ({ label, name, value, onChange, type = "text", placeholder, icon: Icon }) => (
  <div className="form-control">
    <label className="label">
      <span className="label-text">{label}</span>
    </label>
    <div className="relative">
      {Icon && (
        <Icon className="absolute top-1/2 transform -translate-y-1/2 left-3 size-5 text-base-content opacity-70" />
      )}
      <input
        type={type}
        name={name}
        value={value}
        onChange={onChange}
        className={`input input-bordered w-full ${Icon ? 'pl-10' : ''}`}
        placeholder={placeholder}
      />
    </div>
  </div>
);

const TextAreaField = ({ label, name, value, onChange, placeholder }) => (
  <div className="form-control">
    <label className="label">
      <span className="label-text">{label}</span>
    </label>
    <textarea
      name={name}
      value={value}
      onChange={onChange}
      className="textarea textarea-bordered h-24"
      placeholder={placeholder}
    />
  </div>
);

const SubmitButton = ({ isLoading }) => (
  <button className="btn btn-primary w-full" disabled={isLoading} type="submit">
    {!isLoading ? (
      <>
        <ShipWheelIcon className="size-5 mr-2" />
        Complete Onboarding
      </>
    ) : (
      <>
        <LoaderIcon className="animate-spin size-5 mr-2" />
        Onboarding...
      </>
    )}
  </button>
);

const OnboardingPage = () => {
  const { authUser } = useAuthUser();
  const queryClient = useQueryClient();

  const initialFormData = {
    fullName: authUser?.fullName || "",
    bio: authUser?.bio || "",
    location: authUser?.location || "",
    profilePic: authUser?.profilePic || "",
  };

  const [profileData, setProfileData] = useState(initialFormData);

  const onboardMutation = useMutation({
    mutationFn: completeOnboarding,
    onSuccess: () => {
      toast.success(TOAST_MESSAGES.SUCCESS);
      queryClient.invalidateQueries({ queryKey: ["authUser"] });
    },
    onError: (error) => {
      toast.error(error.response.data.message);
    },
  });

  const updateFormField = (field, value) => {
    setProfileData(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleFormSubmission = (e) => {
    e.preventDefault();
    onboardMutation.mutate(profileData);
  };

  const generateNewAvatar = () => {
    const newAvatarUrl = generateRandomAvatarUrl();
    updateFormField("profilePic", newAvatarUrl);
    toast.success(TOAST_MESSAGES.AVATAR_GENERATED);
  };

  return (
    <div className="min-h-screen bg-base-100 flex items-center justify-center p-4">
      <div className="card bg-base-200 w-full max-w-3xl shadow-xl">
        <div className="card-body p-6 sm:p-8">
          <h1 className="text-2xl sm:text-3xl font-bold text-center mb-6">Complete Your Profile</h1>

          <form onSubmit={handleFormSubmission} className="space-y-6">
            <AvatarSection
              currentAvatar={profileData.profilePic}
              onGenerateRandom={generateNewAvatar}
            />

            <FormField
              label="Full Name"
              name="fullName"
              value={profileData.fullName}
              onChange={(e) => updateFormField("fullName", e.target.value)}
              placeholder="Your full name"
            />

            <TextAreaField
              label="Bio"
              name="bio"
              value={profileData.bio}
              onChange={(e) => updateFormField("bio", e.target.value)}
              placeholder="Tell others about yourself and your language learning goals"
            />

            <FormField
              label="Location"
              name="location"
              value={profileData.location}
              onChange={(e) => updateFormField("location", e.target.value)}
              placeholder="City, Country"
              icon={MapPinIcon}
            />

            <SubmitButton isLoading={onboardMutation.isPending} />
          </form>
        </div>
      </div>
    </div>
  );
};

export default OnboardingPage;