import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { ShieldCheck, ShieldAlert, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { PredictionResult } from "@/lib/api";

interface PredictionResultProps {
  result: PredictionResult;
}

export const PredictionResultCard = ({ result }: PredictionResultProps) => {
  const [showSteps, setShowSteps] = useState(false);
  const [animatedConfidence, setAnimatedConfidence] = useState(0);

  const isReal = result.prediction === "REAL";

  // Animate confidence bar
  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedConfidence(result.confidence);
    }, 300);
    return () => clearTimeout(timer);
  }, [result.confidence]);

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in-up">
      {/* Main Verdict Card */}
      <Card
        id="prediction-verdict"
        className={`overflow-hidden border-2 shadow-hard backdrop-blur-sm ${
          isReal
            ? "border-emerald-500/40 bg-emerald-500/5"
            : "border-red-500/40 bg-red-500/5"
        }`}
      >
        {/* Verdict Header */}
        <div
          className={`p-6 md:p-8 relative overflow-hidden ${
            isReal
              ? "bg-gradient-to-r from-emerald-600 via-emerald-500 to-teal-500"
              : "bg-gradient-to-r from-red-600 via-red-500 to-orange-500"
          }`}
        >
          <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-transparent" />
          <div className="flex items-center gap-4 relative z-10">
            <div className="p-3 rounded-2xl bg-white/20 backdrop-blur-sm shadow-lg animate-scale-in">
              {isReal ? (
                <ShieldCheck className="w-10 h-10 text-white" />
              ) : (
                <ShieldAlert className="w-10 h-10 text-white" />
              )}
            </div>
            <div>
              <h2 className="text-2xl md:text-3xl font-display font-black text-white">
                {isReal ? "News Verified — Real ✅" : "News Flagged — Fake 🚨"}
              </h2>
              <p className="text-white/80 mt-1 text-sm md:text-base">
                {isReal
                  ? "This article appears to be from a legitimate source"
                  : "This article shows patterns commonly associated with misinformation"}
              </p>
            </div>
          </div>
        </div>

        {/* Confidence Score */}
        <div className="p-6 md:p-8 space-y-6">
          {/* Probability Bars */}
          <div className="space-y-4">
            <h3 className="text-lg font-display font-bold text-foreground">
              Prediction Confidence
            </h3>

            {/* Real probability */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm font-medium">
                <span className="text-emerald-600 dark:text-emerald-400 flex items-center gap-1.5">
                  <ShieldCheck className="w-4 h-4" /> Real News
                </span>
                <span className="font-mono font-bold">{result.probability_real}%</span>
              </div>
              <div className="h-4 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 rounded-full transition-all duration-1000 ease-out"
                  style={{ width: `${animatedConfidence > 0 ? result.probability_real : 0}%` }}
                />
              </div>
            </div>

            {/* Fake probability */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm font-medium">
                <span className="text-red-500 flex items-center gap-1.5">
                  <ShieldAlert className="w-4 h-4" /> Fake News
                </span>
                <span className="font-mono font-bold">{result.probability_fake}%</span>
              </div>
              <div className="h-4 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-red-500 to-orange-400 rounded-full transition-all duration-1000 ease-out"
                  style={{ width: `${animatedConfidence > 0 ? result.probability_fake : 0}%` }}
                />
              </div>
            </div>

            {/* Overall Confidence */}
            <div className="mt-4 p-4 rounded-xl bg-muted/50 border-2 border-border">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-muted-foreground">Model Confidence</span>
                <span className={`text-2xl font-mono font-black ${
                  isReal ? "text-emerald-600 dark:text-emerald-400" : "text-red-500"
                }`}>
                  {result.confidence}%
                </span>
              </div>
            </div>
          </div>

          {/* Model Info */}
          <div className="flex flex-wrap gap-3 text-xs">
            <span className="px-3 py-1.5 rounded-lg bg-primary/10 text-primary font-semibold border border-primary/20">
              Model: {result.model_used === "lstm_trained" ? "LSTM (Trained)" : "Demo Mode"}
            </span>
            {result.note && (
              <span className="px-3 py-1.5 rounded-lg bg-amber-500/10 text-amber-600 font-semibold border border-amber-500/20">
                ⚠️ {result.note}
              </span>
            )}
          </div>
        </div>
      </Card>

      {/* Preprocessing Steps (Collapsible) */}
      {result.preprocessing_steps && (
        <Card className="overflow-hidden border-2 border-primary/20 shadow-medium backdrop-blur-sm bg-card/80">
          <Button
            id="toggle-preprocessing"
            variant="ghost"
            onClick={() => setShowSteps(!showSteps)}
            className="w-full p-6 flex items-center justify-between hover:bg-primary/5 rounded-none h-auto"
          >
            <span className="text-lg font-display font-bold text-foreground">
              🔬 NLP Preprocessing Steps
            </span>
            {showSteps ? (
              <ChevronUp className="w-5 h-5 text-muted-foreground" />
            ) : (
              <ChevronDown className="w-5 h-5 text-muted-foreground" />
            )}
          </Button>

          {showSteps && (
            <div className="px-6 pb-6 space-y-4 animate-fade-in-up">
              <div className="grid gap-3">
                {[
                  { label: "1. Original Text", value: result.preprocessing_steps.original, color: "text-muted-foreground" },
                  { label: "2. After Cleaning", value: result.preprocessing_steps.after_cleaning, color: "text-blue-600 dark:text-blue-400" },
                  { label: "3. Tokenization", value: result.preprocessing_steps.tokens?.join(", "), color: "text-purple-600 dark:text-purple-400" },
                  { label: "4. Stop-word Removal", value: result.preprocessing_steps.after_stopword_removal?.join(", "), color: "text-orange-600 dark:text-orange-400" },
                  { label: "5. Lemmatization", value: result.preprocessing_steps.after_lemmatization?.join(", "), color: "text-emerald-600 dark:text-emerald-400" },
                ].map((step, i) => (
                  <div key={i} className="p-3 rounded-lg bg-muted/50 border border-border">
                    <span className={`text-sm font-bold ${step.color}`}>{step.label}</span>
                    <p className="text-sm text-foreground/70 mt-1 font-mono break-all line-clamp-3">
                      {step.value || "N/A"}
                    </p>
                  </div>
                ))}
              </div>

              <div className="flex gap-4 text-sm p-3 rounded-lg bg-primary/5 border border-primary/10">
                <span className="text-muted-foreground">
                  Words: <strong className="text-foreground">{result.preprocessing_steps.original_word_count}</strong> →{" "}
                  <strong className="text-primary">{result.preprocessing_steps.final_word_count}</strong>
                </span>
                <span className="text-muted-foreground">
                  Removed: <strong className="text-destructive">{result.preprocessing_steps.words_removed}</strong>
                </span>
              </div>
            </div>
          )}
        </Card>
      )}
    </div>
  );
};
