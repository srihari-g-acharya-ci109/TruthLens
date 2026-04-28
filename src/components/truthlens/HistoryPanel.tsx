import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { History, ShieldCheck, ShieldAlert, Trash2 } from "lucide-react";

interface HistoryEntry {
  id: string;
  text: string;
  prediction: "REAL" | "FAKE";
  confidence: number;
  timestamp: string;
}

const STORAGE_KEY = "truthlens_history";

export function addToHistory(text: string, prediction: "REAL" | "FAKE", confidence: number) {
  const entries = getHistory();
  const entry: HistoryEntry = {
    id: Date.now().toString(),
    text: text.substring(0, 200),
    prediction,
    confidence,
    timestamp: new Date().toLocaleString(),
  };
  entries.unshift(entry);
  // Keep last 20 entries
  const trimmed = entries.slice(0, 20);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
}

export function getHistory(): HistoryEntry[] {
  try {
    const data = localStorage.getItem(STORAGE_KEY);
    return data ? JSON.parse(data) : [];
  } catch {
    return [];
  }
}

export const HistoryPanel = () => {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);

  useEffect(() => {
    setEntries(getHistory());
  }, []);

  const clearHistory = () => {
    localStorage.removeItem(STORAGE_KEY);
    setEntries([]);
  };

  if (entries.length === 0) return null;

  return (
    <div className="max-w-4xl mx-auto space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-display font-bold text-foreground flex items-center gap-2">
          <History className="w-5 h-5 text-primary" />
          Recent Predictions
        </h3>
        <Button
          variant="ghost"
          size="sm"
          onClick={clearHistory}
          className="text-muted-foreground hover:text-destructive"
        >
          <Trash2 className="w-4 h-4 mr-1" />
          Clear
        </Button>
      </div>

      <div className="grid gap-3">
        {entries.slice(0, 5).map((entry) => (
          <Card
            key={entry.id}
            className={`p-4 flex items-center gap-4 border-2 transition-all hover:-translate-y-0.5 bg-card/80 backdrop-blur-sm ${
              entry.prediction === "REAL"
                ? "border-emerald-500/20 hover:border-emerald-500/40"
                : "border-red-500/20 hover:border-red-500/40"
            }`}
          >
            <div
              className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 ${
                entry.prediction === "REAL"
                  ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400"
                  : "bg-red-500/15 text-red-500"
              }`}
            >
              {entry.prediction === "REAL" ? (
                <ShieldCheck className="w-5 h-5" />
              ) : (
                <ShieldAlert className="w-5 h-5" />
              )}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm text-foreground/80 truncate">{entry.text}</p>
              <p className="text-xs text-muted-foreground mt-0.5">{entry.timestamp}</p>
            </div>
            <div className="flex-shrink-0 text-right">
              <div
                className={`text-sm font-bold ${
                  entry.prediction === "REAL"
                    ? "text-emerald-600 dark:text-emerald-400"
                    : "text-red-500"
                }`}
              >
                {entry.prediction}
              </div>
              <div className="text-xs text-muted-foreground font-mono">
                {entry.confidence}%
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
};
