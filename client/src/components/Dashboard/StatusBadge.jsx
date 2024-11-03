
import {    Car,
    AlertTriangle,
    CheckCircle,}
    from "lucide-react";
// Status Badge Component
const StatusBadge = ({ status }) => {
    const getStatusConfig = (status) => {
      switch (status) {
        case "low":
          return { color: "green", icon: CheckCircle };
        case "medium":
          return { color: "yellow", icon: AlertTriangle };
        case "high":
          return { color: "red", icon: AlertTriangle };
        default:
          return { color: "gray", icon: AlertTriangle };
      }
    };
  
    const { color, icon: Icon } = getStatusConfig(status);
  
    return (
      <span
        className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-${color}-100 text-${color}-800`}
      >
        <Icon className={`-ml-0.5 mr-1.5 h-4 w-4 text-${color}-400`} />
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    );
  };

export default StatusBadge;