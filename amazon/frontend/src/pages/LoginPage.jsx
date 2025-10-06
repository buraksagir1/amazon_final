import { useState } from "react";
import { ShipWheelIcon } from "lucide-react";
import { Link } from "react-router";
import useLogin from "../hooks/useLogin";

const THEME = "night";
const BRAND_NAME = "Hearmind";

const FORM_FIELDS = {
  EMAIL: "email",
  PASSWORD: "password",
};

const FormInput = ({ label, type, placeholder, value, onChange, required = true }) => (
  <div className="form-control w-full space-y-2">
    <label className="label">
      <span className="label-text">{label}</span>
    </label>
    <input
      type={type}
      placeholder={placeholder}
      className="input input-bordered w-full"
      value={value}
      onChange={onChange}
      required={required}
    />
  </div>
);

const ErrorAlert = ({ message }) => {
  if (!message) return null;

  return (
    <div className="alert alert-error mb-4">
      <span>{message}</span>
    </div>
  );
};

const SubmitButton = ({ isLoading }) => (
  <button type="submit" className="btn btn-primary w-full" disabled={isLoading}>
    {isLoading ? (
      <>
        <span className="loading loading-spinner loading-xs"></span>
        Signing in...
      </>
    ) : (
      "Sign In"
    )}
  </button>
);

const BrandHeader = () => (
  <div className="mb-4 flex items-center justify-start gap-2">
    <span className="text-3xl font-bold font-mono bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary tracking-wider">
      {BRAND_NAME}
    </span>
  </div>
);

const FormHeader = () => (
  <div>
    <h2 className="text-xl font-semibold">Welcome Back</h2>
    <p className="text-sm opacity-70">
      Sign in to your account to continue your language journey
    </p>
  </div>
);

const SignupPrompt = () => (
  <div className="text-center mt-4">
    <p className="text-sm">
      Don't have an account?{" "}
      <Link to="/signup" className="text-primary hover:underline">
        Create one
      </Link>
    </p>
  </div>
);

const IllustrationSection = () => (
  <div className="hidden lg:flex w-full lg:w-1/2 bg-primary/10 items-center justify-center">
    <div className="max-w-md p-8">
      <div className="relative aspect-square max-w-sm mx-auto">
        <img src="/i.png" alt="Language connection illustration" className="w-full h-full" />
      </div>

      <div className="text-center space-y-3 mt-6">
        <h2 className="text-xl font-semibold">Connect with language partners worldwide</h2>
        <p className="opacity-70">
          Practice conversations, make friends, and improve your language skills together
        </p>
      </div>
    </div>
  </div>
);

const LoginPage = () => {
  const [credentials, setCredentials] = useState({
    [FORM_FIELDS.EMAIL]: "",
    [FORM_FIELDS.PASSWORD]: "",
  });

  const { isPending, error, loginMutation } = useLogin();

  const updateField = (field, value) => {
    setCredentials(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleFormSubmit = (e) => {
    e.preventDefault();
    loginMutation(credentials);
  };

  const errorMessage = error?.response?.data?.message;

  return (
    <div
      className="h-screen flex items-center justify-center p-4 sm:p-6 md:p-8"
      data-theme={THEME}
    >
      <div className="border border-primary/25 flex flex-col lg:flex-row w-full max-w-5xl mx-auto bg-base-100 rounded-xl shadow-lg overflow-hidden">
        <div className="w-full lg:w-1/2 p-4 sm:p-8 flex flex-col">
          <BrandHeader />
          <ErrorAlert message={errorMessage} />

          <div className="w-full">
            <form onSubmit={handleFormSubmit}>
              <div className="space-y-4">
                <FormHeader />

                <div className="flex flex-col gap-3">
                  <FormInput
                    label="Email"
                    type="email"
                    placeholder="hello@example.com"
                    value={credentials[FORM_FIELDS.EMAIL]}
                    onChange={(e) => updateField(FORM_FIELDS.EMAIL, e.target.value)}
                  />

                  <FormInput
                    label="Password"
                    type="password"
                    placeholder="••••••••"
                    value={credentials[FORM_FIELDS.PASSWORD]}
                    onChange={(e) => updateField(FORM_FIELDS.PASSWORD, e.target.value)}
                  />

                  <SubmitButton isLoading={isPending} />
                  <SignupPrompt />
                </div>
              </div>
            </form>
          </div>
        </div>

        <IllustrationSection />
      </div>
    </div>
  );
};

export default LoginPage;