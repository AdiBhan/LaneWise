import StatusBadge from "./StatusBadge";

const MetricsTable = ({ data }) => (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden">
      <div className="p-6 border-b">
        <h3 className="text-lg font-semibold">Detailed Metrics</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Lane
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Status
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Score
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Vehicles
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Speed
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Space
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {data.map((lane) => (
              <tr key={lane.lane_id}>
                <td className="px-6 py-4 whitespace-nowrap">
                  Lane {lane.lane_id}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <StatusBadge status={lane.congestion_level} />
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {lane.score.toFixed(2)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {lane.metrics.vehicle_count}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {Math.round(lane.metrics.average_speed)} mph
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {Math.round(lane.metrics.space_headway)} ft
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

export default MetricsTable;