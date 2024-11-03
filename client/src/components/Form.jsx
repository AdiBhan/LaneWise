import { useEffect, useState } from "react";
import { Car, Gauge, Sliders, Clock, Users } from "lucide-react";

const DataEntryForm = ({ formData, setFormData, onSubmit } ) => {


  useEffect(() => {
    console.log(formData);
  }, [formData]);

  const handleInputChange = (index, field, value) => {
    const newFormData = formData.map((lane, idx) =>
      idx === index ? { ...lane, [field]: Number(value) } : lane
    );
    setFormData(newFormData);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await onSubmit(formData);
      // Optionally, reset form data if needed
      // setFormData(initialFormData);
    } catch (error) {
      console.error("Error submitting form:", error);
    }
  };

  const getGradientColor = (value, max) => {
    const percentage = (value / max) * 100;
    return `linear-gradient(90deg, #4f46e5 ${percentage}%, #e5e7eb ${percentage}%)`;
  };

  // Split lanes into pairs for 2x2 layout
  const laneRows = [
    [formData[0], formData[1]],
    [formData[2], formData[3]],
  ];

  return (
    <div className="max-w-6xl mx-auto p-8">
      <div className="bg-white rounded-2xl shadow-xl p-8 transition-all duration-300 hover:shadow-2xl">
        <div className="flex items-center justify-center gap-3 mb-8">
          <Sliders className="w-8 h-8 text-indigo-600" />
          <h3 className="text-3xl font-bold text-gray-800">Lane Configuration</h3>
        </div>

        <form onSubmit={handleSubmit} className="space-y-8">
          {laneRows.map((row, rowIndex) => (
            <div key={rowIndex} className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {row.map((lane) => (
                <div
                  key={lane.lane_id}
                  className="bg-gray-50 rounded-xl p-6 transition-all duration-300 hover:shadow-md"
                >
                  <div className="flex items-center gap-2 mb-6">
                    <Car className="w-6 h-6 text-indigo-600" />
                    <h4 className="text-xl font-semibold text-gray-800">Lane {lane.lane_id}</h4>
                  </div>

                  <div className="space-y-6">
                    {/* Vehicle Count */}
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Users className="w-4 h-4 text-gray-600" />
                          <label className="text-sm font-medium text-gray-600">Vehicle Count</label>
                        </div>
                        <span className="text-sm font-bold text-indigo-600">{lane.vehicle_count}</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="50"
                        value={lane.vehicle_count}
                        onChange={(e) =>
                          handleInputChange(lane.lane_id - 1, "vehicle_count", e.target.value)
                        }
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                        style={{ background: getGradientColor(lane.vehicle_count, 50) }}
                      />
                    </div>

                    {/* Average Speed */}
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Gauge className="w-4 h-4 text-gray-600" />
                          <label className="text-sm font-medium text-gray-600">Average Speed</label>
                        </div>
                        <span className="text-sm font-bold text-indigo-600">{lane.average_speed} mph</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={lane.average_speed}
                        onChange={(e) =>
                          handleInputChange(lane.lane_id - 1, "average_speed", e.target.value)
                        }
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                        style={{ background: getGradientColor(lane.average_speed, 100) }}
                      />
                    </div>

                    {/* Space Headway */}
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Clock className="w-4 h-4 text-gray-600" />
                          <label className="text-sm font-medium text-gray-600">Space Headway</label>
                        </div>
                        <span className="text-sm font-bold text-indigo-600">{lane.space_headway} ft</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="200"
                        value={lane.space_headway}
                        onChange={(e) =>
                          handleInputChange(lane.lane_id - 1, "space_headway", e.target.value)
                        }
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                        style={{ background: getGradientColor(lane.space_headway, 200) }}
                      />
                    </div>

                    {/* Average Time */}
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Clock className="w-4 h-4 text-gray-600" />
                          <label className="text-sm font-medium text-gray-600">Average Time</label>
                        </div>
                        <span className="text-sm font-bold text-indigo-600">{lane.average_time} ms</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={lane.average_time}
                        onChange={(e) =>
                          handleInputChange(lane.lane_id - 1, "average_time", e.target.value)
                        }
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                        style={{ background: getGradientColor(lane.average_time, 100) }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ))}

          <div className="flex justify-center pt-6">
            <button
              type="submit"
              className="group relative inline-flex items-center justify-center px-8 py-3 font-bold text-white bg-indigo-600 rounded-xl overflow-hidden transition-all duration-300 ease-out hover:bg-indigo-700 hover:scale-105 active:scale-95"
            >
              <span className="absolute inset-0 w-full h-full transition-all duration-300 ease-out transform translate-x-full group-hover:translate-x-0 bg-gradient-to-r from-indigo-500 to-indigo-700"></span>
              <span className="relative flex items-center gap-2">
                Get Recommendation
                <Gauge className="w-5 h-5" />
              </span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default DataEntryForm;
