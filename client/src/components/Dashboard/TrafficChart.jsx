import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
  } from "recharts";

 // Traffic Chart
  const TrafficChart = ({ data }) => (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <h3 className="text-lg font-semibold mb-4">Traffic Flow Analysis</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="lane_id"
              name="Lane"
              label={{ value: "Lane ID", position: "insideBottom", offset: -5 }}
            />
            <YAxis
              yAxisId="left"
              name="Vehicles"
              label={{
                value: "Number of Vehicles",
                angle: -90,
                position: "insideLeft",
                style: { textAnchor: "middle" },
              }}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              name="Speed"
              label={{
                value: "Average Speed (mph)",
                angle: 90,
                position: "insideRight",
                style: { textAnchor: "middle" },
              }}
            />
            <Tooltip />
            <Legend />
            <Bar
              yAxisId="left"
              dataKey="metrics.vehicle_count"
              name="Vehicles"
              fill="#4f46e5"
            />
            <Bar
              yAxisId="right"
              dataKey="metrics.average_speed"
              name="Speed (mph)"
              fill="#10b981"
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
export default TrafficChart;