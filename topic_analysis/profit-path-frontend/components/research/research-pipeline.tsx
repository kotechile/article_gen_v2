"use client";

import { CheckCircle2, Circle, Loader2, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

export type PipelineStep = {
    id: string;
    label: string;
    description: string;
    status: "pending" | "processing" | "complete" | "failed";
};

interface ResearchPipelineProps {
    steps: PipelineStep[];
}

export function ResearchPipeline({ steps }: ResearchPipelineProps) {
    return (
        <div className="w-full py-4">
            <div className="flex items-center justify-between">
                {steps.map((step, index) => (
                    <div key={step.id} className="flex items-center flex-1">
                        <div className="flex flex-col items-center flex-1">
                            {/* Icon */}
                            <div className="relative flex items-center justify-center">
                                {step.status === "pending" && (
                                    <Circle className="h-8 w-8 text-muted-foreground" />
                                )}
                                {step.status === "processing" && (
                                    <Loader2 className="h-8 w-8 text-blue-500 animate-spin" />
                                )}
                                {step.status === "complete" && (
                                    <CheckCircle2 className="h-8 w-8 text-green-500" />
                                )}
                                {step.status === "failed" && (
                                    <XCircle className="h-8 w-8 text-red-500" />
                                )}
                            </div>

                            {/* Label */}
                            <div className="mt-2 text-center">
                                <p
                                    className={cn(
                                        "text-sm font-medium",
                                        step.status === "complete" && "text-green-600",
                                        step.status === "processing" && "text-blue-600",
                                        step.status === "failed" && "text-red-600",
                                        step.status === "pending" && "text-muted-foreground"
                                    )}
                                >
                                    {step.label}
                                </p>
                                <p className="text-xs text-muted-foreground mt-1">
                                    {step.description}
                                </p>
                            </div>
                        </div>

                        {/* Connector Line */}
                        {index < steps.length - 1 && (
                            <div className="flex-1 h-0.5 mx-2 -translate-y-6">
                                <div
                                    className={cn(
                                        "h-full",
                                        steps[index + 1].status !== "pending"
                                            ? "bg-green-500"
                                            : "bg-gray-300"
                                    )}
                                />
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}

// Example usage with default pipeline steps
export const DEFAULT_PIPELINE_STEPS: PipelineStep[] = [
    {
        id: "decomposition",
        label: "Decomposition",
        description: "Generating Seeds",
        status: "pending",
    },
    {
        id: "mining",
        label: "Mining",
        description: "Fetching DataForSEO",
        status: "pending",
    },
    {
        id: "filtering",
        label: "Filtering",
        description: "Applying Thresholds",
        status: "pending",
    },
    {
        id: "clustering",
        label: "Clustering",
        description: "Grouping Keywords",
        status: "pending",
    },
    {
        id: "verification",
        label: "Verification",
        description: "Offers & Trends",
        status: "pending",
    },
];
