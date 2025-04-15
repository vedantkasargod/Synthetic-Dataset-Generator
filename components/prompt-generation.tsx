// components/prompt-generation.tsx

"use client";

import { useState, useEffect } from "react"; // Added useEffect
import axios from 'axios'; // Added axios
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Download, Loader2, Sparkles, Sliders } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

// Define Backend URL
const LOCAL_BACKEND_API_URL = 'http://localhost:8000';
const SDXL_ENDPOINT = '/generate-sdxl/'; // Matches the endpoint in your FastAPI backend

export default function PromptGeneration() {
  const [prompt, setPrompt] = useState("Detailed photo of a defect: small jagged hole in red woolen cloth, studio lighting"); // Example default
  const [negativePrompt, setNegativePrompt] = useState("blurry, low quality, unrealistic, drawing, illustration, text, words"); // Example default
  const [baseImages, setBaseImages] = useState(1);
  const [augmentations, setAugmentations] = useState(4);
  const [inferenceSteps, setInferenceSteps] = useState(30);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [status, setStatus] = useState("Enter prompt and click Generate."); // Initial status
  const [isGenerating, setIsGenerating] = useState(false);
  // const [isComplete, setIsComplete] = useState(false); // downloadUrl indicates completion
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null); // State for blob URL

  // Clean up blob URL when component unmounts or downloadUrl changes
  useEffect(() => {
    const currentUrl = downloadUrl; // Capture current value
    return () => {
      if (currentUrl) {
        console.log("Revoking Blob URL:", currentUrl);
        window.URL.revokeObjectURL(currentUrl);
      }
    };
  }, [downloadUrl]); // Dependency array ensures cleanup runs when URL changes/is removed

  const handleGenerate = async () => {
    if (!prompt) {
        setStatus("⚠️ Please enter a prompt.");
        return;
    }
    if (isGenerating) return; // Prevent multiple clicks

    setIsGenerating(true);
    setStatus("Sending request to backend...");
    setDownloadUrl(null); // Clear previous download link

    const requestData = {
      prompt: prompt,
      negative_prompt: negativePrompt,
      num_base_images: baseImages, // Match backend Pydantic model field name
      augmentations_per_image: augmentations, // Match backend Pydantic model field name
      num_inference_steps: inferenceSteps,
      guidance_scale: guidanceScale,
    };

    try {
      const response = await axios.post(
        `${LOCAL_BACKEND_API_URL}${SDXL_ENDPOINT}`,
        requestData,
        {
          responseType: 'blob', // Expect binary data (zip file)
          timeout: 1800000, // 30 minutes timeout (adjust as needed)
          onDownloadProgress: (progressEvent) => {
              // You could try updating status here, but it's tricky for POST
              // Maybe just indicate receiving data if progressEvent.lengthComputable
              if (status === "Sending request to backend...") { // Update only once
                   setStatus("Receiving response from backend. Sit Tight");
              }
          }
        }
      );

      // Check if response is actually a zip file based on headers
      if (response.headers['content-type'] === 'application/zip') {
          // Create a URL for the blob object
          const blob = new Blob([response.data], { type: 'application/zip' });
          const url = window.URL.createObjectURL(blob);
          setDownloadUrl(url); // Set state to enable download button
          setStatus("Generation complete! Click below to download.");
      } else {
           // If backend returned an error as non-zip (e.g., JSON or text)
           throw new Error("Backend did not return a zip file. Check backend logs.");
      }


    } catch (error: any) {
      console.error("API Error:", error);
      let errorMessage = 'Generation failed. Check backend logs.';
      if (error.response) { // Error response from server (4xx, 5xx)
        // Try to read error message from backend response if possible
         try {
            // Assuming the backend sends JSON error details for HTTPExceptions
            // Need to read blob data as text first
            const errorText = await (error.response.data as Blob).text();
            try {
                const errorJson = JSON.parse(errorText);
                errorMessage = `❌ Error ${error.response.status}: ${errorJson.detail || 'Backend error'}`;
            } catch { // If response wasn't JSON
                 errorMessage = `❌ Error ${error.response.status}: ${errorText.substring(0, 200)}`; // Show first part of text
            }
         } catch (readError) { // If reading blob as text fails
             errorMessage = `❌ Error ${error.response.status}: Could not read backend error response.`;
         }
      } else if (error.request) { // Request made but no response received
        errorMessage = '❌ Network Error: Could not reach backend. Is it running and accessible at ' + LOCAL_BACKEND_API_URL + '? Check CORS.';
      } else { // Other errors (e.g., setting up the request)
         errorMessage = `❌ Request Error: ${error.message}`;
      }
       setStatus(errorMessage);
       setDownloadUrl(null); // Ensure no download button on error
    } finally {
      setIsGenerating(false); // Stop loading indicator regardless of success/failure
    }
  };

  // --- JSX Structure (Use components generated by v0) ---
  return (
    <div className="space-y-8">
      <Card className="bg-neutral-800 border-neutral-700 text-neutral-100">
        {/* --- CardHeader --- */}
        <CardHeader className="pb-3">
          <CardTitle className="text-xl md:text-2xl font-light flex items-center">
            <Sparkles className="w-5 h-5 mr-2 text-teal-500" />
            Prompt-Based Image Generation
          </CardTitle>
          <CardDescription>Generate images using SDXL + LoRA models with custom prompts</CardDescription>
        </CardHeader>

        {/* --- CardContent --- */}
        <CardContent className="space-y-6">
          {/* Prompt Inputs */}
          <div className="space-y-4">
            <div>
              <label htmlFor="prompt" className="block text-sm font-medium mb-2 text-neutral-300">
                Prompt
              </label>
              <Textarea
                id="prompt"
                placeholder="Describe the image you want to generate..."
                rows={4}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className="w-full bg-neutral-700 border-neutral-600 focus:border-teal-500 focus:ring-teal-500 placeholder:text-neutral-500 transition-colors text-neutral-100" // Added text color
              />
            </div>
            <div>
              <label htmlFor="negativePrompt" className="block text-sm font-medium mb-2 text-neutral-300">
                Negative Prompt
              </label>
              <Textarea
                id="negativePrompt"
                placeholder="Elements to avoid in the generated image..."
                rows={3}
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
                className="w-full bg-neutral-700 border-neutral-600 focus:border-teal-500 focus:ring-teal-500 placeholder:text-neutral-500 transition-colors text-neutral-100" // Added text color
              />
            </div>
          </div>

          {/* Parameters */}
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-neutral-200">Parameters</h3>
            {/* Base Images */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <label htmlFor="baseImages" className="block text-sm font-medium text-neutral-300">
                  Number of Base Images: {baseImages}
                </label>
                <div className="flex items-center gap-4">
                  <Slider
                    id="baseImages" min={1} max={10} step={1} value={[baseImages]}
                    onValueChange={(value) => setBaseImages(value[0])}
                    className="flex-1 [&>span:first-child]:h-1 [&>span:first-child>span]:bg-teal-500 [&>span:last-child]:bg-neutral-600" // Added styling
                  />
                  <Input
                    type="number" min={1} max={10} value={baseImages}
                    onChange={(e) => setBaseImages(Math.max(1, Math.min(10, Number(e.target.value) || 1)))} // Added validation
                    className="w-16 bg-neutral-700 border-neutral-600 text-neutral-100"
                  />
                </div>
              </div>
              {/* Augmentations */}
              <div className="space-y-2">
                <label htmlFor="augmentations" className="block text-sm font-medium text-neutral-300">
                  Augmentations per Image: {augmentations}
                  {augmentations === 0 && " (Base only)"}
                </label>
                 <div className="flex items-center gap-4">
                  <Slider
                    id="augmentations" min={0} max={15} step={1} value={[augmentations]}
                    onValueChange={(value) => setAugmentations(value[0])}
                    className="flex-1 [&>span:first-child]:h-1 [&>span:first-child>span]:bg-teal-500 [&>span:last-child]:bg-neutral-600" // Added styling
                  />
                  <Input
                    type="number" min={0} max={15} value={augmentations}
                    onChange={(e) => setAugmentations(Math.max(0, Math.min(15, Number(e.target.value) || 0)))} // Added validation
                    className="w-16 bg-neutral-700 border-neutral-600 text-neutral-100"
                  />
                 </div>
              </div>
            </div>
            {/* Advanced Settings Accordion */}
            <Accordion type="single" collapsible className="w-full">
               <AccordionItem value="advanced" className="border-neutral-700">
                 <AccordionTrigger className="text-neutral-300 hover:text-neutral-100 text-sm py-2"> {/* Adjusted size/padding */}
                   <div className="flex items-center">
                     <Sliders className="w-4 h-4 mr-2" />
                     Advanced Settings
                   </div>
                 </AccordionTrigger>
                 <AccordionContent>
                   <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4"> {/* Added padding */}
                     {/* Inference Steps */}
                     <div className="space-y-2">
                       <label htmlFor="inferenceSteps" className="block text-sm font-medium text-neutral-300">
                         Inference Steps: {inferenceSteps}
                       </label>
                       <div className="flex items-center gap-4">
                         <Slider
                            id="inferenceSteps" min={10} max={75} step={1} value={[inferenceSteps]}
                            onValueChange={(value) => setInferenceSteps(value[0])}
                            className="flex-1 [&>span:first-child]:h-1 [&>span:first-child>span]:bg-teal-500 [&>span:last-child]:bg-neutral-600" // Added styling
                         />
                         <Input
                            type="number" min={10} max={75} value={inferenceSteps}
                            onChange={(e) => setInferenceSteps(Math.max(10, Math.min(75, Number(e.target.value) || 10)))} // Added validation
                            className="w-16 bg-neutral-700 border-neutral-600 text-neutral-100"
                         />
                        </div>
                     </div>
                     {/* Guidance Scale */}
                     <div className="space-y-2">
                       <label htmlFor="guidanceScale" className="block text-sm font-medium text-neutral-300">
                         Guidance Scale: {guidanceScale.toFixed(1)}
                       </label>
                        <div className="flex items-center gap-4">
                         <Slider
                            id="guidanceScale" min={1.0} max={15.0} step={0.1} value={[guidanceScale]}
                            onValueChange={(value) => setGuidanceScale(value[0])}
                            className="flex-1 [&>span:first-child]:h-1 [&>span:first-child>span]:bg-teal-500 [&>span:last-child]:bg-neutral-600" // Added styling
                         />
                         <Input
                            type="number" min={1.0} max={15.0} step={0.1} value={guidanceScale}
                            onChange={(e) => setGuidanceScale(Math.max(1.0, Math.min(15.0, Number(e.target.value) || 1.0)))} // Added validation
                            className="w-16 bg-neutral-700 border-neutral-600 text-neutral-100"
                          />
                         </div>
                     </div>
                   </div>
                 </AccordionContent>
               </AccordionItem>
             </Accordion>
          </div>

          {/* Generate Button */}
          <Button
            onClick={handleGenerate}
            disabled={isGenerating || !prompt.trim()} // Disable if generating or prompt is empty/whitespace
            className="w-full md:w-auto bg-teal-600 hover:bg-teal-500 text-white disabled:opacity-50 disabled:cursor-not-allowed" // Added disabled styles
          >
            {isGenerating ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Sparkles className="mr-2 h-4 w-4" />
                Generate Images
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Status and Download Area */}
      {(status && status !== "Enter prompt and click Generate.") && ( // Only show card if status is not initial
        <Card className="bg-neutral-800 border-neutral-700">
          <CardHeader className="pb-3 pt-4"> {/* Adjusted padding */}
            <CardTitle className="text-lg font-light text-neutral-200">Generation Status</CardTitle>
          </CardHeader>
          <CardContent>
            {/* Display status message with appropriate color */}
            <p className={`text-sm ${status.startsWith('❌') || status.startsWith('⚠️') ? 'text-red-400' : status.startsWith('✅') ? 'text-green-400' : 'text-neutral-300'}`}>
              {status}
            </p>

            {/* Download Button - Use an <a> tag for direct download */}
            {downloadUrl && (
              <a href={downloadUrl} download="sdxl_generated_images.zip" className="inline-block mt-4">
                <Button
                  variant="outline"
                  className="border-teal-700 text-teal-500 hover:bg-teal-900/20 hover:text-teal-400 w-full md:w-auto" // Adjusted width
                >
                  <Download className="mr-2 h-4 w-4" />
                  Download Results (ZIP)
                </Button>
              </a>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}