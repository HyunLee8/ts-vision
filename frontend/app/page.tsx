"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from "recharts";

const ssimData = [
  { epoch: 1, run1: 0.3412, run2: 0.3398 },
  { epoch: 2, run1: 0.3521, run2: 0.3507 },
  { epoch: 3, run1: 0.3589, run2: 0.3571 },
  { epoch: 4, run1: 0.3601, run2: 0.3612 },
  { epoch: 5, run1: 0.3612, run2: 0.3658 },
  { epoch: 10, run1: 0.3628, run2: 0.3652 },
  { epoch: 15, run1: 0.3635, run2: 0.3649 },
  { epoch: 20, run1: 0.3641, run2: 0.3645 },
  { epoch: 30, run1: 0.3657, run2: 0.3638 },
  { epoch: 40, run1: 0.3648, run2: 0.3631 },
  { epoch: 50, run1: null, run2: 0.3625 },
  { epoch: 60, run1: null, run2: 0.3620 },
];

const psnrData = [
  { epoch: 1, run1: 12.81, run2: 12.74 },
  { epoch: 2, run1: 13.15, run2: 13.08 },
  { epoch: 3, run1: 13.28, run2: 13.35 },
  { epoch: 4, run1: 13.35, run2: 13.51 },
  { epoch: 5, run1: 13.39, run2: 13.64 },
  { epoch: 10, run1: 13.45, run2: 13.59 },
  { epoch: 15, run1: 13.50, run2: 13.55 },
  { epoch: 20, run1: 13.55, run2: 13.51 },
  { epoch: 30, run1: 13.64, run2: 13.46 },
  { epoch: 40, run1: 13.58, run2: 13.41 },
  { epoch: 50, run1: null, run2: 13.37 },
  { epoch: 60, run1: null, run2: 13.33 },
];

const disparityData = [
  { epoch: 1, loss: 0.0482 },
  { epoch: 2, loss: 0.0391 },
  { epoch: 3, loss: 0.0345 },
  { epoch: 5, loss: 0.0312 },
  { epoch: 10, loss: 0.0287 },
  { epoch: 15, loss: 0.0271 },
  { epoch: 20, loss: 0.0258 },
  { epoch: 30, loss: 0.0241 },
  { epoch: 40, loss: 0.0229 },
  { epoch: 50, loss: 0.0218 },
  { epoch: 60, loss: 0.0209 },
];

const lossComparison = [
  { name: "L1", run1: 0.0891, run2: 0.0823 },
  { name: "VGG Perceptual", run1: 0.1245, run2: 0.1187 },
  { name: "SSIM", run1: 0.6343, run2: 0.6342 },
  { name: "Disparity", run1: 0.0312, run2: 0.0209 },
  { name: "Cycle", run1: 0.0478, run2: 0.0445 },
  { name: "Smoothness", run1: 0.0034, run2: 0.0031 },
];

const TOTAL = 4;
const DURATIONS = [5000, 12000, 12000, 12000];

const fade = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" } },
  exit: { opacity: 0, y: -10, transition: { duration: 0.4, ease: "easeIn" } },
};

const stagger = {
  animate: { transition: { staggerChildren: 0.12 } },
};

const child = {
  initial: { opacity: 0, y: 16 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" } },
};

function LoopVideo({ src, className }: { src: string; className?: string }) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const v = ref.current;
    if (!v) return;
    const restart = () => { v.currentTime = 0; v.play(); };
    v.addEventListener("ended", restart);
    return () => v.removeEventListener("ended", restart);
  }, []);
  return <video ref={ref} src={src} autoPlay loop muted playsInline className={className} />;
}

function ProgressBar({ active }: { active: number }) {
  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 flex h-1">
      {Array.from({ length: TOTAL }).map((_, i) => (
        <div key={i} className="flex-1 bg-gray-100">
          {i === active && (
            <motion.div
              className="h-full bg-gray-900"
              initial={{ width: "0%" }}
              animate={{ width: "100%" }}
              transition={{ duration: DURATIONS[i] / 1000, ease: "linear" }}
              key={`bar-${active}`}
            />
          )}
          {i < active && <div className="h-full w-full bg-gray-900" />}
        </div>
      ))}
    </div>
  );
}

function Slide1() {
  return (
    <motion.div variants={stagger} initial="initial" animate="animate" exit="exit" className="flex items-center justify-center h-full">
      <div className="max-w-3xl px-8">
        <motion.p variants={child} className="text-xs uppercase tracking-[0.25em] text-gray-400 mb-6">
          University of Virginia — 2026
        </motion.p>
        <motion.h1 variants={child} className="text-6xl font-bold text-gray-900 mb-6 leading-tight">
          Treeshrew<br />Stereo Vision
        </motion.h1>
        <motion.div variants={child} className="w-16 h-px bg-gray-300 mb-6" />
        <motion.p variants={child} className="text-xl text-gray-500 max-w-lg leading-relaxed">
          Monocular to stereoscopic video synthesis for treeshrew neuroscience
          research. MiDaS depth estimation, learned disparity, and ResNet18
          refinement.
        </motion.p>
        <motion.p variants={child} className="text-sm text-gray-400 mt-8">
          Isaac Lee, Benjamin Merkel, Cameron Smith, James Younts
        </motion.p>
      </div>
    </motion.div>
  );
}

function Slide2() {
  return (
    <motion.div variants={stagger} initial="initial" animate="animate" exit="exit" className="flex items-center justify-center h-full">
      <div className="max-w-5xl w-full px-8">
        <motion.p variants={child} className="text-xs uppercase tracking-[0.25em] text-gray-400 mb-3">
          Stereo Output
        </motion.p>
        <motion.h2 variants={child} className="text-4xl font-bold text-gray-900 mb-8">
          Synthesized Stereo Pair
        </motion.h2>
        <motion.div variants={child} className="grid grid-cols-2 gap-6">
          <div>
            <div className="rounded-xl border border-gray-200 shadow-sm overflow-hidden">
              <LoopVideo src="/right_out.mp4" className="w-full" />
            </div>
            <p className="text-sm text-gray-500 mt-3 text-center tracking-wide">Left Eye</p>
          </div>
          <div>
            <div className="rounded-xl border border-gray-200 shadow-sm overflow-hidden">
              <LoopVideo src="/left_out.mp4" className="w-full" />
            </div>
            <p className="text-sm text-gray-500 mt-3 text-center tracking-wide">Right Eye</p>
          </div>
        </motion.div>
        <motion.p variants={child} className="text-sm text-gray-400 mt-6 max-w-2xl">
          Disparity vectors show pixel shift from original. Green arrows on
          the left eye, orange on the right — pointing inward toward the
          convergence axis.
        </motion.p>
      </div>
    </motion.div>
  );
}

function Slide3() {
  return (
    <motion.div variants={stagger} initial="initial" animate="animate" exit="exit" className="flex items-center justify-center h-full">
      <div className="max-w-5xl w-full px-8">
        <motion.p variants={child} className="text-xs uppercase tracking-[0.25em] text-gray-400 mb-3">
          Validation Metrics
        </motion.p>
        <motion.h2 variants={child} className="text-4xl font-bold text-gray-900 mb-8">
          Training Performance
        </motion.h2>
        <motion.div variants={child} className="grid grid-cols-2 gap-6">
          <div className="rounded-xl border border-gray-200 shadow-sm p-6">
            <p className="text-xs uppercase tracking-widest text-gray-400 mb-4">SSIM</p>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={ssimData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
                <YAxis domain={[0.33, 0.37]} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Line type="monotone" dataKey="run1" stroke="#171717" strokeWidth={2} dot={{ r: 2 }} name="Run 1" connectNulls={false} />
                <Line type="monotone" dataKey="run2" stroke="#9ca3af" strokeWidth={2} dot={{ r: 2 }} name="Run 2" />
              </LineChart>
            </ResponsiveContainer>
            <div className="flex gap-4 mt-3 text-xs text-gray-400">
              <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-gray-900 inline-block" />Run 1 — 0.3657 @ ep 30</span>
              <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-gray-400 inline-block" />Run 2 — 0.3658 @ ep 5</span>
            </div>
          </div>
          <div className="rounded-xl border border-gray-200 shadow-sm p-6">
            <p className="text-xs uppercase tracking-widest text-gray-400 mb-4">PSNR (dB)</p>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={psnrData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
                <YAxis domain={[12.5, 14.0]} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Line type="monotone" dataKey="run1" stroke="#171717" strokeWidth={2} dot={{ r: 2 }} name="Run 1" connectNulls={false} />
                <Line type="monotone" dataKey="run2" stroke="#9ca3af" strokeWidth={2} dot={{ r: 2 }} name="Run 2" />
              </LineChart>
            </ResponsiveContainer>
            <div className="flex gap-4 mt-3 text-xs text-gray-400">
              <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-gray-900 inline-block" />Run 1 — 13.64 dB @ ep 30</span>
              <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-gray-400 inline-block" />Run 2 — 13.64 dB @ ep 5</span>
            </div>
          </div>
        </motion.div>
        <motion.p variants={child} className="text-sm text-gray-400 mt-6 max-w-2xl">
          Both runs plateau at nearly identical SSIM and PSNR, indicating an
          architectural ceiling rather than a hyperparameter issue.
        </motion.p>
      </div>
    </motion.div>
  );
}

function Slide4() {
  return (
    <motion.div variants={stagger} initial="initial" animate="animate" exit="exit" className="flex items-center justify-center h-full">
      <div className="max-w-5xl w-full px-8">
        <motion.p variants={child} className="text-xs uppercase tracking-[0.25em] text-gray-400 mb-3">
          Analysis
        </motion.p>
        <motion.h2 variants={child} className="text-4xl font-bold text-gray-900 mb-8">
          Loss Breakdown & Configuration
        </motion.h2>
        <motion.div variants={child} className="grid grid-cols-2 gap-6">
          <div className="rounded-xl border border-gray-200 shadow-sm p-6">
            <p className="text-xs uppercase tracking-widest text-gray-400 mb-4">Loss Components</p>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={lossComparison} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="name" type="category" width={110} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="run1" name="Run 1" barSize={10}>
                  {lossComparison.map((_, i) => <Cell key={i} fill="#171717" />)}
                </Bar>
                <Bar dataKey="run2" name="Run 2" barSize={10}>
                  {lossComparison.map((_, i) => <Cell key={i} fill="#d1d5db" />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="flex gap-4 mt-3 text-xs text-gray-400">
              <span className="flex items-center gap-1.5"><span className="w-3 h-2.5 bg-gray-900 rounded-sm inline-block" />Run 1</span>
              <span className="flex items-center gap-1.5"><span className="w-3 h-2.5 bg-gray-300 rounded-sm inline-block" />Run 2</span>
            </div>
          </div>
          <div className="rounded-xl border border-gray-200 shadow-sm overflow-hidden">
            <p className="text-xs uppercase tracking-widest text-gray-400 px-6 pt-6 mb-4">Training Config</p>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-100">
                  <th className="text-left px-6 py-2 font-medium text-gray-400 text-xs">Parameter</th>
                  <th className="text-left px-6 py-2 font-medium text-gray-400 text-xs">Run 1</th>
                  <th className="text-left px-6 py-2 font-medium text-gray-400 text-xs">Run 2</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                <tr><td className="px-6 py-2.5 text-gray-900">Learning Rate</td><td className="px-6 py-2.5 text-gray-500">1e-4</td><td className="px-6 py-2.5 text-gray-500">5e-5</td></tr>
                <tr><td className="px-6 py-2.5 text-gray-900">Refine Channels</td><td className="px-6 py-2.5 text-gray-500">16</td><td className="px-6 py-2.5 text-gray-500">32</td></tr>
                <tr><td className="px-6 py-2.5 text-gray-900">Epochs</td><td className="px-6 py-2.5 text-gray-500">40</td><td className="px-6 py-2.5 text-gray-500">60</td></tr>
                <tr><td className="px-6 py-2.5 text-gray-900">Crop Size</td><td className="px-6 py-2.5 text-gray-500">256 x 448</td><td className="px-6 py-2.5 text-gray-500">320 x 576</td></tr>
                <tr><td className="px-6 py-2.5 text-gray-900">Best SSIM</td><td className="px-6 py-2.5 font-medium text-gray-900">0.3657</td><td className="px-6 py-2.5 font-medium text-gray-900">0.3658</td></tr>
                <tr><td className="px-6 py-2.5 text-gray-900">Best PSNR</td><td className="px-6 py-2.5 font-medium text-gray-900">13.64 dB</td><td className="px-6 py-2.5 font-medium text-gray-900">13.64 dB</td></tr>
              </tbody>
            </table>
          </div>
        </motion.div>
        <motion.p variants={child} className="text-xs text-gray-300 mt-10 text-center">
          Treeshrew Stereo Vision — University of Virginia — 2026
        </motion.p>
      </div>
    </motion.div>
  );
}

const slides = [Slide1, Slide2, Slide3, Slide4];

export default function Home() {
  const [active, setActive] = useState(0);
  const [paused, setPaused] = useState(false);

  useEffect(() => {
    if (paused) return;
    const t = setTimeout(() => setActive((a) => (a + 1) % TOTAL), DURATIONS[active]);
    return () => clearTimeout(t);
  }, [active, paused]);

  const goTo = useCallback((i: number) => {
    setActive(i);
    setPaused(true);
    setTimeout(() => setPaused(false), DURATIONS[i]);
  }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight" || e.key === " ") { e.preventDefault(); goTo((active + 1) % TOTAL); }
      if (e.key === "ArrowLeft") { e.preventDefault(); goTo((active - 1 + TOTAL) % TOTAL); }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [active, goTo]);

  const Current = slides[active];

  return (
    <div className="h-screen w-screen bg-white relative overflow-hidden">
      <AnimatePresence mode="wait">
        <motion.div key={active} variants={fade} initial="initial" animate="animate" exit="exit" className="absolute inset-0">
          <Current />
        </motion.div>
      </AnimatePresence>
      <ProgressBar active={active} />
      <div className="fixed top-6 right-8 z-50 flex gap-2">
        {Array.from({ length: TOTAL }).map((_, i) => (
          <button
            key={i}
            onClick={() => goTo(i)}
            className={`w-2 h-2 rounded-full transition-all duration-300 ${
              i === active ? "bg-gray-900 scale-125" : "bg-gray-300"
            }`}
          />
        ))}
      </div>
    </div>
  );
}
