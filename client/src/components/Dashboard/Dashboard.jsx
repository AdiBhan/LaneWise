
  import {
    ArrowRight,
  } from "lucide-react";
import LaneCard from "./LaneCard";
import MetricsTable from "./MetricTable";
import TrafficChart from "./TrafficChart";
// Dashboard Component
const Dashboard = ({ recommendations }) => {
    if (!recommendations?.length) return null;
  
    const getBestLane = () => {
      return recommendations.reduce(
        (best, current) => (current.score > (best?.score || 0) ? current : best),
        null
      );
    };
  
    const bestLane = getBestLane();
  
    return (
      <div className="max-w-7xl mx-auto p-6">
        {/* Best Lane Recommendation */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-gray-800">
                Recommended Lane
              </h2>
              <p className="text-gray-600">Based on I-80 traffic conditions</p>
            </div>
            <div className="flex items-center space-x-2 bg-green-100 px-4 py-2 rounded-full">
              <span className="text-xl font-bold text-green-800">
                Lane {bestLane.lane_id}
              </span>
              <ArrowRight className="w-5 h-5 text-green-800" />
            </div>
          </div>
        </div>
        {/* Detailed Metrics Table */}
        <div className="mb-8 gap-6">
          <MetricsTable data={recommendations} />
        </div>
        {/* Lane Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {recommendations.map((lane) => (
            <LaneCard key={lane.lane_id} lane={lane} />
          ))}
        </div>
  
        {/* Traffic Analysis Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-1 gap-6 mb-8">
          <TrafficChart data={recommendations} />
        </div>
      </div>
    );
  };

export default Dashboard