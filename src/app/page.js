"use client";
import { useState } from "react";
import axios from "axios";

export default function Home() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="flex flex-col items-center p-8">
      <h1 className="text-3xl font-bold mb-4">🌱 Crop Disease Prediction</h1>
      <form onSubmit={handleSubmit} className="flex flex-col items-center space-y-4">
        <input
          type="file"
          onChange={(e) => setFile(e.target.files[0])}
          className="border p-2"
        />
        <button type="submit" className="bg-green-500 text-white px-4 py-2 rounded">
          Predict
        </button>
      </form>
      {result && (
        <div className="mt-6 p-4 border rounded shadow">
          <h2 className="text-xl font-semibold">Disease: {result.disease}</h2>
          <p className="mt-2">Solution: {result.solution}</p>
        </div>
      )}
    </div>
  );
}
