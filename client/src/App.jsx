import React, { useEffect } from "react";
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
import {
  Car,
  AlertTriangle,
  CheckCircle,
  ArrowRight,
  Gauge,
  Sliders,
  Clock,
  Users,
  Video,
  ClipboardList,
  BarChart2
} from "lucide-react";
import { useState } from "react";
import Header from "./components/Header";
import Footer from "./components/Footer";
import Dashboard from "./components/Dashboard/Dashboard";
import DataEntryForm from "./components/Form";

export default function LaneWiseApp() {
  const initialFormData = [
    { lane_id: 1, vehicle_count: 10, average_speed: 60, space_headway: 100, average_time: 50 },
    { lane_id: 2, vehicle_count: 15, average_speed: 55, space_headway: 90, average_time: 40 },
    { lane_id: 3, vehicle_count: 20, average_speed: 50, space_headway: 80, average_time: 70 },
    { lane_id: 4, vehicle_count: 25, average_speed: 45, space_headway: 70, average_time: 20 },
  ];

  const [formData, setFormData] = useState(initialFormData);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState('form');

  const handleUserDataSubmit = async (userData) => {
    console.log(userData);
    const formattedData = userData.map((lane) => ({
      lane_id: lane.lane_id,
      vehicle_count: lane.vehicle_count,
      avg_speed: lane.average_speed,
      avg_space: lane.space_headway,
      avg_time: lane.average_time,
    }));
    
    console.log(formattedData);
    setLoading(true);
    try {
      const response = await fetch("http://localhost:8000/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formattedData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setData(result);
      setError(null);
    } catch (err) {
      console.error("Error sending data to the backend:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const toggleMode = () => {
    setMode(mode === 'form' ? 'realtime' : 'form');
    // Clear data when switching to realtime mode
    if (mode === 'form') {
      setData(null);
    }
  };

  const renderContent = () => {
    if (loading) {
      return (
        <div className="flex items-center justify-center h-64">
          <div className="text-lg text-gray-600">Processing data...</div>
        </div>
      );
    }

    if (error) {
      return (
        <div className="max-w-7xl mx-auto p-6">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <div className="text-red-800">Error: {error}</div>
          </div>
        </div>
      );
    }

    if (mode === 'form') {
      return (
        <>
         <DataEntryForm formData={formData} setFormData={setFormData} onSubmit={handleUserDataSubmit} />
          {data && <Dashboard recommendations={data} />}
        </>
      );
    }

    return (
      <div className="bg-white rounded-lg p-8 shadow-lg">
        <div className="flex flex-col items-center gap-4 text-center">
          <div className="relative">
            <BarChart2 className="w-12 h-12 text-blue-500" />
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
          </div>
          <h2 className="text-xl font-semibold text-gray-800">Real-time Traffic Analysis</h2>
          <p className="text-gray-600 max-w-md">
            Live traffic monitoring and analysis coming soon. 
          </p>
      
        </div>
      </div>
    );
  };
  return (
    <div className="min-h-screen flex flex-col bg-gray-100">
      <Header />
      <div className="flex-grow">
        <div className="max-w-7xl mx-auto p-4">
          <div className="flex justify-center mb-6">
            <div className="bg-white rounded-full shadow-lg p-1">
              <button
                onClick={toggleMode}
                className={`flex items-center gap-3 px-6 py-3 rounded-full transition-all duration-300 ${
                  mode === 'form'
                    ? 'bg-white text-gray-700 hover:bg-gray-50'
                    : 'bg-blue-500 text-white hover:bg-blue-600'
                }`}
              >
                <div className="flex items-center gap-2">
                  {mode === 'form' ? (
                    <>
                      <Video className="w-5 h-5" />
                      <div className="flex flex-col items-start">
                        <span className="text-sm font-semibold">Real-time Analysis</span>
                        <span className="text-xs opacity-75">Switch to live traffic monitoring</span>
                      </div>
                    </>
                  ) : (
                    <>
                      <ClipboardList className="w-5 h-5" />
                      <div className="flex flex-col items-start">
                        <span className="text-sm font-semibold">Manual Entry</span>
                        <span className="text-xs opacity-75">Switch to form input</span>
                      </div>
                    </>
                  )}
                </div>
              </button>
            </div>
          </div>
          {renderContent()}
        </div>
      </div>
      <Footer />
    </div>
  );
}