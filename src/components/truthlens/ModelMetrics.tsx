import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { BarChart3, TrendingUp, Target, Award } from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Cell, Legend
} from "recharts";
import { getModelMetrics, getTrainingHistory } from "@/lib/api";
import type { ModelMetrics as MetricsType, TrainingHistory } from "@/lib/api";

export const ModelMetrics = () => {
  const [metrics, setMetrics] = useState<MetricsType | null>(null);
  const [history, setHistory] = useState<TrainingHistory | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [m, h] = await Promise.all([getModelMetrics(), getTrainingHistory()]);
        setMetrics(m);
        setHistory(h);
      } catch (err) {
        console.error("Failed to fetch metrics:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-spin w-8 h-8 border-4 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  if (!metrics) return null;

  const metricCards = [
    { label: "Accuracy", value: metrics.accuracy, icon: Target, color: "from-blue-600 to-cyan-500" },
    { label: "Precision", value: metrics.precision, icon: Award, color: "from-purple-600 to-pink-500" },
    { label: "Recall", value: metrics.recall, icon: TrendingUp, color: "from-emerald-600 to-teal-500" },
    { label: "F1 Score", value: metrics.f1_score, icon: BarChart3, color: "from-orange-600 to-amber-500" },
  ];

  // Confusion matrix data
  const cm = metrics.confusion_matrix;
  const confusionData = [
    { name: "True Negative", value: cm.true_negative, fill: "#22c55e" },
    { name: "True Positive", value: cm.true_positive, fill: "#3b82f6" },
    { name: "False Positive", value: cm.false_positive, fill: "#f59e0b" },
    { name: "False Negative", value: cm.false_negative, fill: "#ef4444" },
  ];

  // Training history chart data
  const lossData = history?.epochs.map((e, i) => ({
    epoch: e,
    "Train Loss": history.train_loss[i],
    "Val Loss": history.val_loss[i],
  })) || [];

  const accuracyData = history?.epochs.map((e, i) => ({
    epoch: e,
    "Train Accuracy": history.train_accuracy[i],
    "Val Accuracy": history.val_accuracy[i],
  })) || [];

  return (
    <div className="space-y-8">
      {/* Section Header */}
      <div className="text-center space-y-2">
        <h2 className="text-3xl md:text-4xl font-display font-black">
          <span className="bg-gradient-to-r from-primary via-secondary to-accent bg-clip-text text-transparent">
            Model Performance
          </span>
        </h2>
        <p className="text-muted-foreground text-base md:text-lg max-w-2xl mx-auto">
          Evaluation metrics from training on the{" "}
          <span className="font-semibold text-primary">{metrics.dataset}</span>{" "}
          ({metrics.training_samples.toLocaleString()} training samples)
        </p>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {metricCards.map((m, i) => (
          <Card
            key={i}
            className="p-5 md:p-6 text-center space-y-3 border-2 border-primary/10 hover:border-primary/30 hover:shadow-glow-primary transition-all duration-300 hover:-translate-y-1 bg-card/80 backdrop-blur-sm animate-fade-in-up"
            style={{ animationDelay: `${i * 0.1}s` }}
          >
            <div className={`mx-auto w-12 h-12 rounded-2xl bg-gradient-to-br ${m.color} flex items-center justify-center shadow-medium`}>
              <m.icon className="w-6 h-6 text-white" />
            </div>
            <div className="text-3xl md:text-4xl font-mono font-black text-foreground">
              {m.value}%
            </div>
            <div className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">
              {m.label}
            </div>
          </Card>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Loss Curve */}
        <Card className="p-6 border-2 border-primary/10 bg-card/80 backdrop-blur-sm">
          <h3 className="text-lg font-display font-bold text-foreground mb-4">
            📉 Training & Validation Loss
          </h3>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={lossData}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="epoch" stroke="hsl(var(--muted-foreground))" fontSize={12} />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "2px solid hsl(var(--border))",
                  borderRadius: "12px",
                  fontSize: "13px",
                }}
              />
              <Legend />
              <Line type="monotone" dataKey="Train Loss" stroke="#8b5cf6" strokeWidth={2.5} dot={false} />
              <Line type="monotone" dataKey="Val Loss" stroke="#06b6d4" strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        {/* Accuracy Curve */}
        <Card className="p-6 border-2 border-primary/10 bg-card/80 backdrop-blur-sm">
          <h3 className="text-lg font-display font-bold text-foreground mb-4">
            📈 Training & Validation Accuracy
          </h3>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={accuracyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="epoch" stroke="hsl(var(--muted-foreground))" fontSize={12} />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} domain={[50, 100]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "2px solid hsl(var(--border))",
                  borderRadius: "12px",
                  fontSize: "13px",
                }}
              />
              <Legend />
              <Line type="monotone" dataKey="Train Accuracy" stroke="#8b5cf6" strokeWidth={2.5} dot={false} />
              <Line type="monotone" dataKey="Val Accuracy" stroke="#22c55e" strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* Confusion Matrix */}
      <Card className="p-6 border-2 border-primary/10 bg-card/80 backdrop-blur-sm">
        <h3 className="text-lg font-display font-bold text-foreground mb-4">
          🔢 Confusion Matrix
        </h3>
        <div className="grid md:grid-cols-2 gap-6 items-center">
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={confusionData}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={11} />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--card))",
                  border: "2px solid hsl(var(--border))",
                  borderRadius: "12px",
                  fontSize: "13px",
                }}
              />
              <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                {confusionData.map((entry, index) => (
                  <Cell key={index} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          {/* Matrix Grid */}
          <div className="space-y-3">
            <div className="grid grid-cols-3 gap-2 text-center text-sm">
              <div></div>
              <div className="font-bold text-muted-foreground p-2">Pred: Real</div>
              <div className="font-bold text-muted-foreground p-2">Pred: Fake</div>

              <div className="font-bold text-muted-foreground p-2 text-right">Real</div>
              <div className="p-3 rounded-lg bg-emerald-500/15 border-2 border-emerald-500/30 font-mono font-bold text-emerald-600 dark:text-emerald-400 text-lg">
                {cm.true_negative}
              </div>
              <div className="p-3 rounded-lg bg-amber-500/15 border-2 border-amber-500/30 font-mono font-bold text-amber-600 dark:text-amber-400 text-lg">
                {cm.false_positive}
              </div>

              <div className="font-bold text-muted-foreground p-2 text-right">Fake</div>
              <div className="p-3 rounded-lg bg-red-500/15 border-2 border-red-500/30 font-mono font-bold text-red-500 text-lg">
                {cm.false_negative}
              </div>
              <div className="p-3 rounded-lg bg-blue-500/15 border-2 border-blue-500/30 font-mono font-bold text-blue-600 dark:text-blue-400 text-lg">
                {cm.true_positive}
              </div>
            </div>

            <div className="text-xs text-muted-foreground space-y-1 pt-2">
              <p><strong className="text-emerald-500">TN</strong> = Correctly identified real news</p>
              <p><strong className="text-blue-500">TP</strong> = Correctly identified fake news</p>
              <p><strong className="text-amber-500">FP</strong> = Real incorrectly flagged as fake</p>
              <p><strong className="text-red-500">FN</strong> = Fake incorrectly classified as real</p>
            </div>
          </div>
        </div>
      </Card>

      {/* Model Architecture */}
      <Card className="p-6 border-2 border-primary/10 bg-card/80 backdrop-blur-sm">
        <h3 className="text-lg font-display font-bold text-foreground mb-4">
          🏗️ Model Architecture
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {[
            { label: "Architecture", value: metrics.model_architecture.type },
            { label: "Embedding Dim", value: metrics.model_architecture.embedding_dim },
            { label: "Hidden Dim", value: metrics.model_architecture.hidden_dim },
            { label: "LSTM Layers", value: metrics.model_architecture.num_layers },
            { label: "Dropout", value: `${metrics.model_architecture.dropout * 100}%` },
            { label: "Max Seq Length", value: metrics.model_architecture.max_sequence_length },
          ].map((item, i) => (
            <div key={i} className="p-3 rounded-lg bg-muted/50 border border-border text-center">
              <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">
                {item.label}
              </div>
              <div className="text-sm font-bold text-foreground font-mono">{item.value}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
};
