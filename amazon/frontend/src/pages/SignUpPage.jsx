import { useState } from "react";
import { ShipWheelIcon } from "lucide-react";
import { Link } from "react-router";
import useSignUp from "../hooks/useSignUp";

const APP_THEME = "night";
const BRAND_NAME = "Hearmind";
const MIN_PASSWORD_LENGTH = 6;

const FORM_FIELDS = {
  FULL_NAME: "fullName",
  EMAIL: "email",
  PASSWORD: "password",
};

const ErrorDisplay = ({ error }) => {
  if (!error) return null;

  return (
    <div className="alert alert-error mb-4">
      <span>{error.response.data.message}</span>
    </div>
  );
};

const PageHeader = () => (
  <div>
    <h2 className="text-xl font-semibold">Create an Account</h2>
    <p className="text-sm opacity-70">
      Join {BRAND_NAME} and start your language learning adventure!
    </p>
  </div>
);

const InputField = ({ label, type, placeholder, value, onChange, helpText }) => (
  <div className="form-control w-full">
    <label className="label">
      <span className="label-text">{label}</span>
    </label>
    <input
      type={type}
      placeholder={placeholder}
      className="input input-bordered w-full"
      value={value}
      onChange={onChange}
      required
    />
    {helpText && <p className="text-xs opacity-70 mt-1">{helpText}</p>}
  </div>
);

const TermsCheckbox = () => (
  <div className="form-control">
    <label className="label cursor-pointer justify-start gap-2">
      <input type="checkbox" className="checkbox checkbox-sm" required />
      <span className="text-xs leading-tight">
        I agree to the{" "}
        <span className="text-primary hover:underline">terms of service</span> and{" "}
        <span className="text-primary hover:underline">privacy policy</span>
      </span>
    </label>
  </div>
);

const SubmitButton = ({ isSubmitting }) => (
  <button className="btn btn-primary w-full" type="submit">
    {isSubmitting ? (
      <>
        <span className="loading loading-spinner loading-xs"></span>
        Loading...
      </>
    ) : (
      "Create Account"
    )}
  </button>
);

const LoginPrompt = () => (
  <div className="text-center mt-4">
    <p className="text-sm">
      Already have an account?{" "}
      <Link to="/login" className="hover:underline">
        Sign in
      </Link>
    </p>
  </div>
);

const IllustrationPanel = () => (
  <div className="hidden lg:flex w-full lg:w-1/2 bg-primary/10 items-center justify-center">
    <div className="max-w-md p-8">
      <div className="relative aspect-square max-w-sm mx-auto">
        <img src="/s.png" alt="Language connection illustration" className="w-full h-full" />
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

const SignUpPage = () => {
  const [registrationData, setRegistrationData] = useState({
    [FORM_FIELDS.FULL_NAME]: "",
    [FORM_FIELDS.EMAIL]: "",
    [FORM_FIELDS.PASSWORD]: "",
  });

  const { isPending, error, signupMutation } = useSignUp();

  const updateField = (fieldName, fieldValue) => {
    setRegistrationData(prev => ({
      ...prev,
      [fieldName]: fieldValue,
    }));
  };

  const handleFormSubmit = (e) => {
    e.preventDefault();
    signupMutation(registrationData);
  };

  return (
    <div
      className="h-screen flex items-center justify-center p-4 sm:p-6 md:p-8"
      data-theme={APP_THEME}
    >
      <div className="border border-primary/25 flex flex-col lg:flex-row w-full max-w-5xl mx-auto bg-base-100 rounded-xl shadow-lg overflow-hidden">
        <div className="w-full lg:w-1/2 p-4 sm:p-8 flex flex-col">
          <ErrorDisplay error={error} />

          <div className="w-full">
            <form onSubmit={handleFormSubmit}>
              <div className="space-y-4">
                <PageHeader />

                <div className="space-y-3">
                  <InputField
                    label="Full Name"
                    type="text"
                    placeholder="John Doe"
                    value={registrationData[FORM_FIELDS.FULL_NAME]}
                    onChange={(e) => updateField(FORM_FIELDS.FULL_NAME, e.target.value)}
                  />

                  <InputField
                    label="Email"
                    type="email"
                    placeholder="john@gmail.com"
                    value={registrationData[FORM_FIELDS.EMAIL]}
                    onChange={(e) => updateField(FORM_FIELDS.EMAIL, e.target.value)}
                  />

                  <InputField
                    label="Password"
                    type="password"
                    placeholder="********"
                    value={registrationData[FORM_FIELDS.PASSWORD]}
                    onChange={(e) => updateField(FORM_FIELDS.PASSWORD, e.target.value)}
                    helpText={`Password must be at least ${MIN_PASSWORD_LENGTH} characters long`}
                  />

                  <TermsCheckbox />
                </div>

                <SubmitButton isSubmitting={isPending} />
                <LoginPrompt />
              </div>
            </form>
          </div>
        </div>

        <IllustrationPanel />
      </div>
    </div>
  );
};

export default SignUpPage;