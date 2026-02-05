"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Loader2, Save } from "lucide-react";
import { toast } from "sonner";

interface ResearchSettings {
    min_volume: number;
    max_difficulty: number;
    min_cpc: number;
    strict_mode: boolean;
}

import { apiClient } from "@/lib/api-client";

interface SettingsResponse {
    success: boolean;
    data?: ResearchSettings;
    message?: string;
}

export default function ResearchSettingsPage() {
    // const { toast } = useToast(); -> Removed
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [settings, setSettings] = useState<ResearchSettings>({
        min_volume: 50,
        max_difficulty: 50,
        min_cpc: 0.5,
        strict_mode: true,
    });

    useEffect(() => {
        fetchSettings();
    }, []);

    const fetchSettings = async () => {
        try {
            const result = await apiClient.get<SettingsResponse>("/settings/research");

            if (result.success && result.data) {
                setSettings(result.data);
            }
        } catch (error) {
            console.error("Failed to fetch settings:", error);
            toast.error("Error", {
                description: "Failed to load settings. Using defaults.",
            });
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        setSaving(true);
        try {
            const result = await apiClient.post<SettingsResponse>("/settings/research", settings);

            if (result.success) {
                toast.success("Success", {
                    description: "Research settings saved successfully",
                });
            } else {
                throw new Error(result.message || "Failed to save settings");
            }
        } catch (error) {
            console.error("Failed to save settings:", error);
            toast.error("Error", {
                description: "Failed to save settings",
            });
        } finally {
            setSaving(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <Loader2 className="h-8 w-8 animate-spin" />
            </div>
        );
    }

    return (
        <div className="container mx-auto py-8 max-w-4xl">
            <div className="mb-8">
                <h1 className="text-3xl font-bold">Research Settings</h1>
                <p className="text-muted-foreground mt-2">
                    Configure thresholds for ProfitPath keyword filtering
                </p>
            </div>

            <Card>
                <CardHeader>
                    <CardTitle>ProfitPath Filtering Thresholds</CardTitle>
                    <CardDescription>
                        These settings control how aggressively the system filters keywords during research.
                        Strict mode will completely remove unprofitable keywords, while non-strict mode will flag them.
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                    <div className="space-y-2">
                        <Label htmlFor="min_volume">Minimum Search Volume</Label>
                        <Input
                            id="min_volume"
                            type="number"
                            value={settings.min_volume}
                            onChange={(e) =>
                                setSettings({ ...settings, min_volume: parseInt(e.target.value) || 0 })
                            }
                            min="0"
                        />
                        <p className="text-sm text-muted-foreground">
                            Keywords with search volume below this threshold will be filtered out (Spec default: 50)
                        </p>
                    </div>

                    <div className="space-y-2">
                        <Label htmlFor="max_difficulty">Maximum Keyword Difficulty (KD)</Label>
                        <Input
                            id="max_difficulty"
                            type="number"
                            value={settings.max_difficulty}
                            onChange={(e) =>
                                setSettings({ ...settings, max_difficulty: parseInt(e.target.value) || 0 })
                            }
                            min="0"
                            max="100"
                        />
                        <p className="text-sm text-muted-foreground">
                            Keywords with difficulty above this threshold will be filtered out (Spec default: 50)
                        </p>
                    </div>

                    <div className="space-y-2">
                        <Label htmlFor="min_cpc">Minimum CPC ($)</Label>
                        <Input
                            id="min_cpc"
                            type="number"
                            step="0.1"
                            value={settings.min_cpc}
                            onChange={(e) =>
                                setSettings({ ...settings, min_cpc: parseFloat(e.target.value) || 0 })
                            }
                            min="0"
                        />
                        <p className="text-sm text-muted-foreground">
                            Minimum cost-per-click value for commercial viability (Spec default: 0.5)
                        </p>
                    </div>

                    <div className="flex items-center justify-between">
                        <div className="space-y-0.5">
                            <Label htmlFor="strict_mode">Strict Mode</Label>
                            <p className="text-sm text-muted-foreground">
                                If enabled, unprofitable keywords are ruthlessly deleted. If disabled, they are flagged but kept.
                            </p>
                        </div>
                        <Switch
                            id="strict_mode"
                            checked={settings.strict_mode}
                            onCheckedChange={(checked: boolean) =>
                                setSettings({ ...settings, strict_mode: checked })
                            }
                        />
                    </div>

                    <div className="flex justify-end pt-4">
                        <Button onClick={handleSave} disabled={saving}>
                            {saving ? (
                                <>
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                    Saving...
                                </>
                            ) : (
                                <>
                                    <Save className="mr-2 h-4 w-4" />
                                    Save Settings
                                </>
                            )}
                        </Button>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
