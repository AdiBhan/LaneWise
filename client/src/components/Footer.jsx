const Footer = () => (
    <div className="bg-indigo-600 text-white py-4">
      <div className="max-w-7xl mx-auto text-center">
        <p className="text-sm">
          © {new Date().getFullYear()} Adi Bhan - LaneWise. All rights reserved.
        </p>
        <p className="text-xs mt-1">
          Powered by React and FastAPI |{" "}
          <a
            href="https://github.com/AdiBhan/LaneWise/"
            target="_blank"
            rel="noopener noreferrer"
            className="underline text-gray-200 hover:text-white"
          >
            GitHub Repository
          </a>
        </p>
      </div>
    </div>
  );

export default Footer;