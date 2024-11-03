// Header Component
import {
  Car,
} from "lucide-react";
const Header = () => (
  <div className="bg-indigo-600 text-white p-4">
    <div className="max-w-7xl mx-auto flex items-center justify-between">
      <div className="flex items-center space-x-2">
        <Car className="w-8 h-8" />
        <h1 className="text-2xl font-bold">LaneWise</h1>
      </div>
      <div className="flex items-center space-x-4">
        <span className="px-3 py-1 bg-green-500 rounded-full text-sm">
          System Active
        </span>
        <span className="text-sm">
          Last Updated: {new Date().toLocaleTimeString()}
        </span>
      </div>
    </div>
  </div>
);

export default Header;
