"use client";

import { useEffect, useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "http://127.0.0.1:8000";

const documentTypes = [
  {
    name: "Invoice",
    description: "Bills, invoices, and payment documents",
    icon: "▣",
    accent: "emerald",
  },
  {
    name: "Resume",
    description: "CVs and professional resumes",
    icon: "♟",
    accent: "violet",
  },
  {
    name: "Form",
    description: "Applications and structured forms",
    icon: "⚙",
    accent: "amber",
  },
  {
    name: "Budget",
    description: "Budgets and financial statements",
    icon: "▥",
    accent: "cyan",
  },
  {
    name: "Advertisement",
    description: "Promotional and advertisement documents",
    icon: "◈",
    accent: "pink",
  },
];

const features = [
  {
    title: "AI-Powered Analysis",
    description: "LayoutLMv3 + OCR for document understanding",
    icon: "✦",
  },
  {
    title: "High Accuracy",
    description: "93.42% test accuracy across 5 document classes",
    icon: "◉",
  },
  {
    title: "Fast Inference",
    description: "Optimized inference pipeline for document analysis",
    icon: "⚡",
  },
  {
    title: "Privacy First",
    description: "Documents are processed by your inference service",
    icon: "▣",
  },
];

type PredictionResult = {
  filename: string;
  document_type: string;
  confidence: number;
  ocr_words: number;
  probabilities: Record<string, number>;
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const [result, setResult] =
    useState<PredictionResult | null>(null);

  const [isAnalyzing, setIsAnalyzing] =
    useState(false);

  const [analysisStage, setAnalysisStage] =
    useState("");

  const [error, setError] =
    useState<string | null>(null);

  const [backendOnline, setBackendOnline] =
    useState(false);

  // --------------------------------------------------
  // Backend health monitoring
  // --------------------------------------------------

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch(
          `${API_URL}/health`,
          {
            method: "GET",
            cache: "no-store",
          }
        );

        if (!response.ok) {
          throw new Error("Backend unavailable");
        }

        setBackendOnline(true);
      } catch {
        setBackendOnline(false);
      }
    };

    checkBackend();

    const interval = setInterval(
      checkBackend,
      10000
    );

    return () => {
      clearInterval(interval);
    };
  }, []);

  // --------------------------------------------------
  // File handling
  // --------------------------------------------------

  const handleFile = (selectedFile: File) => {
    setError(null);
    setResult(null);

    if (
      selectedFile.type !== "image/png" &&
      selectedFile.type !== "image/jpeg"
    ) {
      setError(
        "Please upload a PNG or JPEG image."
      );
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError(
        "File size must be less than 10 MB."
      );
      return;
    }

    setFile(selectedFile);

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    const url =
      URL.createObjectURL(selectedFile);

    setPreview(url);
  };

  const handleInputChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const selectedFile =
      event.target.files?.[0];

    if (selectedFile) {
      handleFile(selectedFile);
    }
  };

  const handleDrop = (
    event: React.DragEvent<HTMLLabelElement>
  ) => {
    event.preventDefault();
    setIsDragging(false);

    const droppedFile =
      event.dataTransfer.files?.[0];

    if (droppedFile) {
      handleFile(droppedFile);
    }
  };

  const removeFile = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setAnalysisStage("");
  };

  // --------------------------------------------------
  // Document prediction
  // --------------------------------------------------

  const analyzeDocument = async () => {
    if (
      !file ||
      isAnalyzing ||
      !backendOnline
    ) {
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResult(null);
    setAnalysisStage(
      "Uploading document..."
    );

    try {
      const formData = new FormData();

      formData.append(
        "file",
        file
      );

      setAnalysisStage(
        "Running OCR and extracting document layout..."
      );

      const response = await fetch(
        `${API_URL}/predict/`,
        {
          method: "POST",
          body: formData,
        }
      );

      setAnalysisStage(
        "Running LayoutLMv3 inference..."
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail ||
            "Document prediction failed."
        );
      }

      setAnalysisStage(
        "Preparing prediction results..."
      );

      setResult(data);
    } catch (err) {
      console.error(err);

      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to the prediction service."
      );

      // Immediately reflect a failed backend request.
      setBackendOnline(false);
    } finally {
      setIsAnalyzing(false);
      setAnalysisStage("");
    }
  };

  return (
    <main className="min-h-screen overflow-x-hidden bg-[#020617] text-white">

      {/* Background glow */}
      <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
        <div className="absolute left-1/4 top-0 h-[500px] w-[500px] rounded-full bg-blue-600/10 blur-[140px]" />

        <div className="absolute right-0 top-1/3 h-[450px] w-[450px] rounded-full bg-violet-600/10 blur-[140px]" />
      </div>

      {/* HEADER */}
      <header className="border-b border-slate-800/80 bg-[#020617]/80 backdrop-blur-xl">
        <div className="mx-auto flex h-20 max-w-7xl items-center justify-between px-6">

          <div className="flex items-center gap-4">

            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-violet-600 shadow-lg shadow-blue-500/20">
              <span className="text-xl font-bold">
                ◈
              </span>
            </div>

            <div>
              <div className="flex items-center gap-3">

                <h1 className="text-xl font-bold tracking-tight">
                  DocIntel
                </h1>

                <span className="hidden h-4 w-px bg-slate-700 sm:block" />

                <span className="hidden text-sm text-slate-400 sm:block">
                  LayoutLMv3 Document Intelligence
                </span>

              </div>
            </div>

          </div>

          <div className="flex items-center gap-4">

            {/* Real backend status */}
            <div
              className={`flex items-center gap-2 rounded-full px-4 py-2 text-sm ${
                backendOnline
                  ? "border border-emerald-500/30 bg-emerald-500/5 text-emerald-400"
                  : "border border-red-500/30 bg-red-500/5 text-red-400"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  backendOnline
                    ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]"
                    : "bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.8)]"
                }`}
              />

              {backendOnline
                ? "Model Online"
                : "Model Offline"}
            </div>

            <a
              href="https://github.com/abhinavv0516/docintel-layoutlmv3"
              target="_blank"
              rel="noreferrer"
              className="hidden h-10 items-center justify-center rounded-full border border-slate-800 px-4 text-sm text-slate-400 transition hover:border-slate-600 hover:text-white sm:flex"
            >
              GitHub
            </a>

          </div>

        </div>
      </header>

      {/* HERO */}
      <section className="mx-auto max-w-7xl px-6 pb-10 pt-16 lg:pt-20">

        <div className="grid items-start gap-12 lg:grid-cols-[0.9fr_1.1fr]">

          {/* LEFT */}
          <div className="pt-4">

            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-blue-500/20 bg-blue-500/5 px-4 py-2 text-sm text-blue-300">
              <span className="h-1.5 w-1.5 rounded-full bg-blue-400" />
              AI-Powered Document Intelligence
            </div>

            <h2 className="max-w-xl text-5xl font-bold leading-[1.08] tracking-tight sm:text-6xl">

              Intelligent Document
              <br />

              Analysis with

              <span className="block bg-gradient-to-r from-blue-400 via-blue-500 to-violet-500 bg-clip-text text-transparent">
                LayoutLMv3
              </span>

            </h2>

            <p className="mt-7 max-w-xl text-lg leading-8 text-slate-400">
              Upload a document and let DocIntel analyze its
              text, visual structure, and layout to determine
              its document type.
            </p>

            {/* Features */}
            <div className="mt-9 space-y-3">

              {features.map((feature) => (
                <div
                  key={feature.title}
                  className="group flex items-center gap-4 rounded-xl border border-slate-800/80 bg-slate-900/40 px-4 py-3.5 transition hover:border-slate-700 hover:bg-slate-900/70"
                >

                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-blue-500/10 text-lg text-blue-400">
                    {feature.icon}
                  </div>

                  <div>

                    <h3 className="text-sm font-semibold">
                      {feature.title}
                    </h3>

                    <p className="mt-0.5 text-sm text-slate-500">
                      {feature.description}
                    </p>

                  </div>

                </div>
              ))}

            </div>

          </div>

          {/* RIGHT */}
          <div>

            {/* Upload card */}
            <div className="overflow-hidden rounded-2xl border border-blue-500/20 bg-slate-900/50 shadow-2xl shadow-blue-950/20 backdrop-blur-xl">

              <label
                onDragOver={(event) => {
                  event.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => {
                  setIsDragging(false);
                }}
                onDrop={handleDrop}
                className={`m-5 flex min-h-[390px] cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed px-6 py-12 text-center transition ${
                  isDragging
                    ? "border-blue-400 bg-blue-500/10"
                    : "border-slate-700 bg-slate-950/40 hover:border-blue-500/60 hover:bg-slate-950/70"
                }`}
              >

                {!file ? (
                  <>
                    <div className="mb-7 flex h-20 w-20 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500/15 to-violet-500/15 text-4xl text-blue-400">
                      ↑
                    </div>

                    <h3 className="text-xl font-semibold">
                      Drag & drop your document here
                    </h3>

                    <p className="mt-3 text-sm text-slate-500">
                      PNG or JPEG images up to 10 MB
                    </p>

                    <div className="my-5 flex w-full max-w-xs items-center gap-3">

                      <div className="h-px flex-1 bg-slate-800" />

                      <span className="text-xs text-slate-600">
                        or
                      </span>

                      <div className="h-px flex-1 bg-slate-800" />

                    </div>

                    <span className="rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 px-8 py-3.5 font-semibold shadow-lg shadow-blue-600/20 transition hover:from-blue-500 hover:to-violet-500">
                      Choose File
                    </span>

                    <input
                      type="file"
                      accept="image/png,image/jpeg"
                      className="hidden"
                      onChange={handleInputChange}
                    />
                  </>
                ) : (

                  <div className="w-full">

                    <div className="mb-5 flex items-center justify-between">

                      <div className="text-left">

                        <p className="font-semibold">
                          Document selected
                        </p>

                        <p className="mt-1 max-w-xs truncate text-sm text-slate-500">
                          {file.name}
                        </p>

                      </div>

                      <button
                        type="button"
                        disabled={isAnalyzing}
                        onClick={(event) => {
                          event.preventDefault();
                          removeFile();
                        }}
                        className="rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-400 transition hover:border-red-500/40 hover:text-red-400 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        Remove
                      </button>

                    </div>

                    {preview && (
                      <div className="flex max-h-[280px] justify-center overflow-hidden rounded-xl border border-slate-800 bg-black/30 p-3">

                        <img
                          src={preview}
                          alt="Document preview"
                          className="max-h-[250px] max-w-full rounded-lg object-contain"
                        />

                      </div>
                    )}

                  </div>

                )}

              </label>

              {/* Analyze button */}
              {file && (
                <div className="px-5 pb-5">

                  <button
                    type="button"
                    onClick={analyzeDocument}
                    disabled={
                      isAnalyzing ||
                      !backendOnline
                    }
                    className="flex w-full items-center justify-center gap-3 rounded-xl bg-gradient-to-r from-blue-600 to-violet-600 px-6 py-4 font-semibold shadow-lg shadow-blue-600/20 transition hover:from-blue-500 hover:to-violet-500 disabled:cursor-not-allowed disabled:opacity-70"
                  >

                    {isAnalyzing && (
                      <span className="h-5 w-5 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                    )}

                    <span>
                      {isAnalyzing
                        ? "Analyzing Document..."
                        : !backendOnline
                        ? "Backend Offline"
                        : "Analyze Document"}
                    </span>

                  </button>

                </div>
              )}

              {/* Inference status */}
              {isAnalyzing && (
                <div className="mx-5 mb-5 rounded-xl border border-blue-500/20 bg-blue-500/5 px-4 py-4">

                  <div className="flex items-center gap-3">

                    <div className="h-2.5 w-2.5 animate-pulse rounded-full bg-blue-400 shadow-[0_0_10px_rgba(96,165,250,0.8)]" />

                    <div>

                      <p className="text-sm font-medium text-blue-300">
                        {analysisStage || "Processing document..."}
                      </p>

                      <p className="mt-1 text-xs text-slate-500">
                        OCR → Layout analysis → LayoutLMv3 classification
                      </p>

                    </div>

                  </div>

                  <div className="mt-3 h-1 overflow-hidden rounded-full bg-slate-800">

                    <div className="h-full w-1/2 animate-pulse rounded-full bg-gradient-to-r from-blue-500 to-violet-500" />

                  </div>

                </div>
              )}

              {/* Error */}
              {error && (
                <div className="mx-5 mb-5 rounded-xl border border-red-500/30 bg-red-500/5 px-4 py-3 text-sm text-red-400">
                  {error}
                </div>
              )}

              {/* Security strip */}
              <div className="flex items-center justify-center gap-2 border-t border-slate-800 px-5 py-4 text-sm text-slate-500">

                <span className="text-emerald-400">
                  ◉
                </span>

                Documents are processed by the inference service

              </div>

            </div>

            {/* Prediction result */}
            {result && (
              <div className="mt-4 overflow-hidden rounded-2xl border border-emerald-500/20 bg-slate-900/60 p-6 shadow-xl shadow-emerald-950/10">

                <div className="mb-6 flex items-center justify-between">

                  <div>

                    <p className="text-xs uppercase tracking-widest text-slate-500">
                      Prediction
                    </p>

                    <h3 className="mt-1 text-3xl font-bold capitalize text-white">
                      {result.document_type}
                    </h3>

                  </div>

                  <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 text-right">

                    <p className="text-2xl font-bold text-emerald-400">
                      {(result.confidence * 100).toFixed(2)}%
                    </p>

                    <p className="text-xs text-slate-500">
                      Confidence
                    </p>

                  </div>

                </div>

                <div className="mb-6 grid grid-cols-2 gap-3">

                  <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">

                    <p className="text-xl font-bold text-blue-400">
                      {result.ocr_words}
                    </p>

                    <p className="mt-1 text-xs text-slate-500">
                      OCR Words
                    </p>

                  </div>

                  <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">

                    <p className="truncate text-sm font-semibold text-slate-300">
                      {result.filename}
                    </p>

                    <p className="mt-1 text-xs text-slate-500">
                      Processed File
                    </p>

                  </div>

                </div>

                {/* Probabilities */}
                <div>

                  <div className="mb-4 flex items-center justify-between">

                    <h4 className="text-sm font-semibold text-slate-300">
                      Class Probabilities
                    </h4>

                    <span className="text-xs text-slate-600">
                      LayoutLMv3
                    </span>

                  </div>

                  <div className="space-y-3">

                    {Object.entries(result.probabilities)
                      .sort(([, a], [, b]) => b - a)
                      .map(([name, probability]) => (

                        <div key={name}>

                          <div className="mb-1 flex justify-between text-xs">

                            <span className="capitalize text-slate-400">
                              {name}
                            </span>

                            <span className="text-slate-500">
                              {(probability * 100).toFixed(2)}%
                            </span>

                          </div>

                          <div className="h-2 overflow-hidden rounded-full bg-slate-800">

                            <div
                              className={`h-full rounded-full transition-all ${
                                name === result.document_type
                                  ? "bg-gradient-to-r from-blue-500 to-violet-500"
                                  : "bg-slate-600"
                              }`}
                              style={{
                                width: `${Math.max(
                                  probability * 100,
                                  1
                                )}%`,
                              }}
                            />

                          </div>

                        </div>

                      ))}

                  </div>

                </div>

                <button
                  type="button"
                  onClick={removeFile}
                  className="mt-6 w-full rounded-xl border border-slate-700 px-5 py-3 text-sm font-semibold text-slate-400 transition hover:border-slate-500 hover:text-white"
                >
                  Analyze Another Document
                </button>

              </div>
            )}

            {/* Stats */}
            {!result && (
              <div className="mt-4 grid grid-cols-3 gap-3">

                <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">

                  <p className="text-2xl font-bold text-blue-400">
                    93.42%
                  </p>

                  <p className="mt-1 text-xs text-slate-500">
                    Test Accuracy
                  </p>

                </div>

                <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">

                  <p className="text-2xl font-bold text-emerald-400">
                    5
                  </p>

                  <p className="mt-1 text-xs text-slate-500">
                    Document Types
                  </p>

                </div>

                <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-4">

                  <p className="text-2xl font-bold text-violet-400">
                    OCR
                  </p>

                  <p className="mt-1 text-xs text-slate-500">
                    Tesseract Engine
                  </p>

                </div>

              </div>
            )}

          </div>

        </div>

      </section>

      {/* DOCUMENT TYPES */}
      <section className="mx-auto max-w-7xl px-6 py-10">

        <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-6 sm:p-8">

          <div className="mb-7 flex items-center gap-3">

            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-500/10 text-blue-400">
              ◈
            </div>

            <div>

              <h2 className="font-semibold">
                Supported Document Types
              </h2>

              <p className="text-sm text-slate-500">
                Classification categories supported by the model
              </p>

            </div>

          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">

            {documentTypes.map((type) => (

              <div
                key={type.name}
                className="group rounded-xl border border-slate-800 bg-slate-950/50 p-5 text-center transition hover:-translate-y-1 hover:border-slate-700 hover:bg-slate-900"
              >

                <div
                  className={`mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl text-xl ${
                    type.accent === "emerald"
                      ? "bg-emerald-500/10 text-emerald-400"
                      : type.accent === "violet"
                      ? "bg-violet-500/10 text-violet-400"
                      : type.accent === "amber"
                      ? "bg-amber-500/10 text-amber-400"
                      : type.accent === "cyan"
                      ? "bg-cyan-500/10 text-cyan-400"
                      : "bg-pink-500/10 text-pink-400"
                  }`}
                >
                  {type.icon}
                </div>

                <h3 className="font-semibold">
                  {type.name}
                </h3>

                <p className="mt-2 text-xs leading-5 text-slate-500">
                  {type.description}
                </p>

              </div>

            ))}

          </div>

        </div>

      </section>

      {/* FOOTER */}
      <footer className="mt-8 border-t border-slate-800">

        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-3 px-6 py-7 text-sm text-slate-500 sm:flex-row">

          <p>
            © 2026 DocIntel
          </p>

          <p>
            Built with{" "}
            <span className="text-slate-300">
              LayoutLMv3
            </span>
            {" "}·{" "}
            <span className="text-slate-300">
              Tesseract OCR
            </span>
            {" "}·{" "}
            <span className="text-slate-300">
              FastAPI
            </span>
          </p>

        </div>

      </footer>

    </main>
  );
}