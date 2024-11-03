import { Gauge } from "lucide-react";

// Lane Status Card
const LaneCard = ({ lane }) => {
  const getStatusColor = (congestion) => {
    switch (congestion) {
      case "low":
        return "bg-green-100 text-green-800 border-green-300";
      case "medium":
        return "bg-yellow-100 text-yellow-800 border-yellow-300";
      case "high":
        return "bg-red-100 text-red-800 border-red-300";
      default:
        return "bg-gray-100 text-gray-800 border-gray-300";
    }
  };

  return (
    <div
      className={`relative p-6 rounded-lg border-2 ${getStatusColor(
        lane.congestion_level
      )}`}
    >
      <div className="flex justify-between items-start">
        <div>
          <h3 className="text-lg font-semibold">Lane {lane.lane_id}</h3>
          <div className="mt-1 flex items-center">
            <Gauge className="w-4 h-4 mr-1" />
            <span className="text-sm capitalize">
              {lane.congestion_level} Congestion
            </span>
          </div>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold">
            {Math.round(lane.metrics.average_speed)}
          </div>
          <div className="text-sm">mph</div>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
        <div>
          <div className="text-gray-600">Vehicles</div>
          <div className="font-semibold">{lane.metrics.vehicle_count}</div>
        </div>
        <div>
          <div className="text-gray-600">Space</div>
          <div className="font-semibold">
            {Math.round(lane.metrics.space_headway)}ft
          </div>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium-bold">Score</span>
          <span className="font-bold">{lane.score.toFixed(2)}</span>
        </div>
        <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-current rounded-full h-2 transition-all duration-500"
            style={{ width: `${lane.score * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
};

export default LaneCard;
