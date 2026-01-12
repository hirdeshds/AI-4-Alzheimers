import { useNavigate } from "react-router-dom";
import { useAuth } from "@/contexts/AuthContext";
import { Button } from "@/components/ui/button";
import { Brain, LogOut, Upload, Activity, FileText, TrendingUp, User, Code, Copy, CheckCircle, Server } from "lucide-react";
import { useState } from "react";

const Dashboard = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [copied, setCopied] = useState(false);

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  const displayName = user?.displayName || user?.email?.split("@")[0] || "User";
  const userEmail = user?.email || "No email";
  const joinDate = "N/A";

  const apiEndpoint = "http://127.0.0.1:8000/predict";
  
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const sampleCode = `fetch("${apiEndpoint}", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ image: "<base64_image>" })
})`;

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md border-b border-border">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <a href="/" className="flex items-center gap-2 group">
              <div className="p-2 rounded-xl gradient-primary">
                <Brain className="h-6 w-6 text-primary-foreground" />
              </div>
              <span className="text-xl font-bold text-foreground">
                AI for <span className="gradient-text">Alzheimer's</span>
              </span>
            </a>

            <div className="flex items-center gap-4">
              <span className="text-muted-foreground hidden sm:block">
                Welcome, <span className="text-foreground font-medium">{displayName}</span>
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={handleLogout}
                className="gap-2"
              >
                <LogOut className="h-4 w-4" />
                <span className="hidden sm:inline">Logout</span>
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="pt-24 pb-12 px-6">
        <div className="container mx-auto max-w-6xl">
          {/* User Profile Card */}
          <div className="bg-card rounded-2xl p-6 border border-border shadow-card mb-8">
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
              <div className="p-4 rounded-full gradient-primary">
                <User className="h-8 w-8 text-primary-foreground" />
              </div>
              <div className="flex-1">
                <h2 className="text-2xl font-bold text-foreground">{displayName}</h2>
                <p className="text-muted-foreground">{userEmail}</p>
                <p className="text-sm text-muted-foreground mt-1">Member since: {joinDate}</p>
              </div>
              <Button 
                variant="outline" 
                onClick={() => navigate("/prediction")}
                className="gap-2"
              >
                <Upload className="h-4 w-4" />
                New Scan
              </Button>
            </div>
          </div>

          <div className="mb-8">
            <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-4">
              Your <span className="gradient-text">Dashboard</span>
            </h1>
            <p className="text-lg text-muted-foreground">
              Access AI-powered Alzheimer's detection tools and manage your analyses.
            </p>
          </div>

          {/* Feature Cards */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Prediction Card */}
            <div
              onClick={() => navigate("/prediction")}
              className="bg-card rounded-2xl p-6 border border-border shadow-card hover:shadow-elevated transition-all cursor-pointer group"
            >
              <div className="p-3 rounded-xl gradient-primary w-fit mb-4 group-hover:scale-110 transition-transform">
                <Upload className="h-6 w-6 text-primary-foreground" />
              </div>
              <h3 className="text-xl font-bold text-foreground mb-2">
                New Prediction
              </h3>
              <p className="text-muted-foreground mb-4">
                Upload an MRI scan for AI-powered Alzheimer's disease detection and analysis.
              </p>
              <Button className="gradient-primary hover:opacity-90 transition-opacity w-full">
                Start Analysis
              </Button>
            </div>

            {/* Activity Card */}
            <div className="bg-card rounded-2xl p-6 border border-border shadow-card">
              <div className="p-3 rounded-xl bg-secondary/10 w-fit mb-4">
                <Activity className="h-6 w-6 text-secondary" />
              </div>
              <h3 className="text-xl font-bold text-foreground mb-2">
                Recent Activity
              </h3>
              <p className="text-muted-foreground mb-4">
                View your recent scan analyses and prediction history.
              </p>
              <div className="text-sm text-muted-foreground italic">
                No recent activity yet
              </div>
            </div>

            {/* Stats Card */}
            <div className="bg-card rounded-2xl p-6 border border-border shadow-card">
              <div className="p-3 rounded-xl bg-accent w-fit mb-4">
                <TrendingUp className="h-6 w-6 text-accent-foreground" />
              </div>
              <h3 className="text-xl font-bold text-foreground mb-2">
                Your Statistics
              </h3>
              <div className="space-y-3 mt-4">
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Total Scans</span>
                  <span className="font-semibold text-foreground">0</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">This Month</span>
                  <span className="font-semibold text-foreground">0</span>
                </div>
              </div>
            </div>

            {/* API Connection Guide */}
            {/* <div className="bg-card rounded-2xl p-6 border border-border shadow-card md:col-span-2 lg:col-span-3">
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-xl gradient-primary w-fit">
                  <Server className="h-6 w-6 text-primary-foreground" />
                </div>
                <div className="flex-1">
                  <h3 className="text-xl font-bold text-foreground mb-2">
                    API Connection Guide
                  </h3>
                  <p className="text-muted-foreground mb-4">
                    Connect your local prediction API to analyze MRI scans. Make sure your backend server is running.
                  </p>
                  
                  <div className="space-y-4">
                    <div>
                      <h4 className="text-sm font-semibold text-foreground mb-2 flex items-center gap-2">
                        <Code className="h-4 w-4" /> API Endpoint
                      </h4>
                      <div className="flex items-center gap-2 bg-muted rounded-lg p-3">
                        <code className="text-sm text-foreground flex-1 break-all">{apiEndpoint}</code>
                        <Button 
                          variant="ghost" 
                          size="sm"
                          onClick={() => copyToClipboard(apiEndpoint)}
                          className="shrink-0"
                        >
                          {copied ? <CheckCircle className="h-4 w-4 text-green-500" /> : <Copy className="h-4 w-4" />}
                        </Button>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-sm font-semibold text-foreground mb-2">Quick Setup Steps:</h4>
                      <ol className="text-sm text-muted-foreground space-y-2 list-decimal list-inside">
                        <li>Ensure your FastAPI/Flask server is running on port 8000</li>
                        <li>The API should accept POST requests with image data</li>
                        <li>Response should include prediction class and confidence scores</li>
                        <li>Use the "New Prediction" feature to upload and analyze scans</li>
                      </ol>
                    </div>

                    <div>
                      <h4 className="text-sm font-semibold text-foreground mb-2">Sample Request:</h4>
                      <pre className="bg-muted rounded-lg p-3 text-xs text-foreground overflow-x-auto">
                        {sampleCode}
                      </pre>
                    </div>
                  </div>
                </div>
              </div>
            </div> */}

            {/* Help Card */}
            <div className="bg-card rounded-2xl p-6 border border-border shadow-card md:col-span-2 lg:col-span-3">
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-xl bg-muted w-fit">
                  <FileText className="h-6 w-6 text-muted-foreground" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-foreground mb-2">
                    How It Works
                  </h3>
                  <p className="text-muted-foreground">
                    Our AI system analyzes MRI brain scans to detect early signs of Alzheimer's disease. 
                    Simply upload a scan, and our model will provide a prediction along with confidence levels 
                    for different stages of cognitive impairment. This tool is meant to assist medical 
                    professionals and should not replace clinical diagnosis.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
