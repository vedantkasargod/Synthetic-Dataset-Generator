// components/overlay-generation.tsx

"use client";

import { useState, useEffect, useCallback } from "react"; // Added useEffect, useCallback
import axios from 'axios'; // Added axios
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Download, Loader2, X, Upload, Layers, ImageIcon, FileSymlink, Settings, Sliders, Trash2 } from "lucide-react"; // Added Trash2
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

// Define Backend URL
const LOCAL_BACKEND_API_URL = 'http://localhost:8000';
const OVERLAY_ENDPOINT = '/generate-overlay/'; // Matches the endpoint in your FastAPI backend

// Interface for category state
type Category = {
  name: string;
  files: File[]; // Store File objects directly
};

export default function OverlayGeneration() {
  // --- State Variables ---
  // File/Category Management
  const [categories, setCategories] = useState<Category[]>([]);
  const [currentCategoryName, setCurrentCategoryName] = useState("");
  const [currentForegroundFiles, setCurrentForegroundFiles] = useState<FileList | null>(null); // From file input
  const [backgroundFiles, setBackgroundFiles] = useState<File[]>([]); // Store File objects
  const [augmentationFile, setAugmentationFile] = useState<File | null>(null);

  // Generation Parameters
  const [numImages, setNumImages] = useState(50); // Default lower for testing
  const [outputWidth, setOutputWidth] = useState(640);
  const [outputHeight, setOutputHeight] = useState(480);
  const [maxObjects, setMaxObjects] = useState(3);
  const [scaleRange, setScaleRange] = useState([0.2, 0.5]); // Store range tuple
  const [avoidOverlap, setAvoidOverlap] = useState(false); // Default off as it's not implemented
  const [useParallel, setUseParallel] = useState(true); // Default based on previous logic

  // UI State
  const [status, setStatus] = useState("Upload foregrounds & backgrounds, configure, then generate.");
  const [isGenerating, setIsGenerating] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);

   // Clean up blob URL
  useEffect(() => {
    const currentUrl = downloadUrl;
    return () => {
      if (currentUrl) {
        window.URL.revokeObjectURL(currentUrl);
        console.log("Revoked Blob URL:", currentUrl);
      }
    };
  }, [downloadUrl]);

  // --- Event Handlers ---
  const handleAddCategory = useCallback(() => {
    const name = currentCategoryName.trim();
    if (name && currentForegroundFiles && currentForegroundFiles.length > 0) {
      // Check if category exists, add files if it does, otherwise create new
      setCategories(prevCategories => {
          const existingCategoryIndex = prevCategories.findIndex(cat => cat.name === name);
          const newFilesArray = Array.from(currentForegroundFiles); // Convert FileList to Array

          if (existingCategoryIndex > -1) {
              // Add files to existing category (avoid duplicates by name)
              const updatedCategories = [...prevCategories];
              const existingFiles = new Set(updatedCategories[existingCategoryIndex].files.map(f => f.name));
              const filesToAdd = newFilesArray.filter(f => !existingFiles.has(f.name));
              updatedCategories[existingCategoryIndex].files.push(...filesToAdd);
              if(filesToAdd.length > 0) console.log(`Added ${filesToAdd.length} files to category '${name}'`);
              return updatedCategories;
          } else {
              // Add new category
               console.log(`Adding new category '${name}' with ${newFilesArray.length} files`);
              return [...prevCategories, { name: name, files: newFilesArray }];
          }
      });
      setCurrentCategoryName(""); // Clear input
      setCurrentForegroundFiles(null); // Clear file input selection state *visually* needs direct DOM manipulation or key change
       // Reset file input visually - simplest way is changing the key
       // This requires adding a key state: const [fgUploaderKey, setFgUploaderKey] = useState(Date.now());
       // and then in button handler: setFgUploaderKey(Date.now()); Add key={fgUploaderKey} to input
    } else if (!name) {
        setStatus("Please enter a category name.");
    } else {
         setStatus("Please select foreground files to add.");
    }
  }, [currentCategoryName, currentForegroundFiles]); // Dependencies for useCallback

  const handleRemoveCategory = (index: number) => {
    setCategories(categories.filter((_, i) => i !== index));
  };

  const handleClearForegrounds = () => {
    setCategories([]);
    setCurrentCategoryName("");
    setCurrentForegroundFiles(null);
    // Reset file input key if using that method
  };

   const handleForegroundFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setCurrentForegroundFiles(e.target.files);
   };

   const handleBackgroundFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            const newFiles = Array.from(e.target.files);
            setBackgroundFiles(prevFiles => {
                const existingNames = new Set(prevFiles.map(f => f.name));
                const filesToAdd = newFiles.filter(f => !existingNames.has(f.name));
                return [...prevFiles, ...filesToAdd];
            });
            // Reset file input visually if needed by changing key
        }
   };

  const handleRemoveBackgroundFile = (index: number) => {
    setBackgroundFiles(prevFiles => prevFiles.filter((_, i) => i !== index));
  };

  const handleClearBackgrounds = () => {
    setBackgroundFiles([]);
  };

   const handleAugmentationFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setAugmentationFile(e.target.files ? e.target.files[0] : null);
   };

  const handleClearAugmentation = () => {
    setAugmentationFile(null);
     // Reset file input visually if needed by changing key
  };


  // --- MAIN GENERATION FUNCTION ---
  const handleGenerate = async () => {
    if (categories.length === 0 || backgroundFiles.length === 0) {
        setStatus("Please upload at least one foreground category AND background images.");
        return;
    }
    if (isGenerating) return;

    setIsGenerating(true);
    setStatus("Preparing data and sending request...");
    setDownloadUrl(null);

    // 1. Create FormData
    const formData = new FormData();

    // 2. Append Files
    // Backgrounds
    backgroundFiles.forEach(file => {
        formData.append('background_files', file, file.name);
    });

    // Foregrounds (with category encoded in filename)
    let fgCount = 0;
    categories.forEach(category => {
        category.files.forEach(file => {
            // Create a filename like 'categoryName__originalFilename.png'
            // Ensure category name is filesystem-safe (replace spaces, special chars if needed)
            const safeCategoryName = category.name.replace(/[^a-zA-Z0-9_-]/g, '_');
            const newFileName = `${safeCategoryName}__${file.name}`;
            formData.append('foreground_files', file, newFileName);
            fgCount++;
        });
    });
     if (fgCount === 0) {
         setStatus("No valid foreground files found in the added categories.");
         setIsGenerating(false);
         return;
     }

    // Augmentation file
    if (augmentationFile) {
        formData.append('augmentation_file', augmentationFile, augmentationFile.name);
    }

    // 3. Append Configuration as JSON String
    const configData = {
        image_number: numImages,
        max_objects_per_image: maxObjects,
        image_width: outputWidth,
        image_height: outputHeight,
        scaling_factors: scaleRange, // Use the state variable holding the array
        avoid_collisions: avoidOverlap,
        parallelize: useParallel,
    };
    formData.append('config_json', JSON.stringify(configData));

    // 4. Make API Call
    try {
      const response = await axios.post(
        `${LOCAL_BACKEND_API_URL}${OVERLAY_ENDPOINT}`,
        formData, // Send FormData
        {
          responseType: 'blob', // Expect binary data (zip file)
          timeout: 1800000, // 30 minutes timeout (adjust as needed)
          headers: {
            // Content-Type is set automatically for FormData by axios/browser
          },
           onUploadProgress: (progressEvent) => {
                // Track upload progress (optional)
                // const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                // setStatus(`Uploading files: ${percentCompleted}%`);
                if (status === "Preparing data and sending request...") {
                    setStatus("Uploading files to backend...");
                }
            },
            onDownloadProgress: (progressEvent) => {
                 if (status === "Uploading files to backend...") { // Or a dedicated uploading status
                    setStatus("Backend processing... Downloading results (can take time)...");
                 }
            }
        }
      );

      // 5. Handle Response (same as prompt generator)
      if (response.headers['content-type'] === 'application/zip') {
          const blob = new Blob([response.data], { type: 'application/zip' });
          const url = window.URL.createObjectURL(blob);
          setDownloadUrl(url);
          setStatus("Dataset generation complete! Click below to download.");
      } else {
           throw new Error("Backend did not return a zip file. Check backend logs.");
      }

    } catch (error: any) {
       console.error("API Error:", error);
       // Same detailed error handling as in prompt generator
       let errorMessage = 'Generation failed. Check backend logs.';
        if (error.response) {
            try {
                const errorText = await (error.response.data as Blob).text();
                try {
                    const errorJson = JSON.parse(errorText);
                    errorMessage = `❌ Error ${error.response.status}: ${errorJson.detail || 'Backend error'}`;
                } catch { errorMessage = `❌ Error ${error.response.status}: ${errorText.substring(0, 200)}`; }
            } catch { errorMessage = `❌ Error ${error.response.status}: Could not read backend error response.`; }
        } else if (error.request) { errorMessage = '❌ Network Error: Could not reach backend. Is it running?'; }
        else { errorMessage = `❌ Request Error: ${error.message}`; }
        setStatus(errorMessage);
        setDownloadUrl(null);
    } finally {
      setIsGenerating(false);
    }
  };


  // --- JSX Structure ---
  return (
     <div className="space-y-8">
      <Card className="bg-neutral-800 border-neutral-700 text-neutral-100">
        <CardHeader className="pb-3">
          <CardTitle className="text-xl md:text-2xl font-light flex items-center">
            <Layers className="w-5 h-5 mr-2 text-teal-500" />
            Overlay Synthetic Dataset Generator
          </CardTitle>
          <CardDescription>
            Upload objects and backgrounds to create custom labeled datasets.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Data Upload Section */}
           <div className="space-y-6">
             <h3 className="text-lg font-medium text-neutral-200 flex items-center">
               <Upload className="w-4 h-4 mr-2 text-teal-500" />
               Data Upload
             </h3>
             <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
               {/* --- Foregrounds Card --- */}
                <Card className="bg-neutral-700 border-neutral-600">
                 <CardHeader className="pb-3">
                   <CardTitle className="text-md font-medium flex items-center">
                     <Layers className="w-4 h-4 mr-2 text-teal-400" /> Foregrounds (PNG)
                   </CardTitle>
                 </CardHeader>
                 <CardContent className="space-y-4">
                   {/* Category Name Input */}
                   <div>
                     <Label htmlFor="categoryName" className="mb-2 block text-neutral-300 text-sm">New Category Name</Label>
                     <Input id="categoryName" value={currentCategoryName} onChange={(e) => setCurrentCategoryName(e.target.value)} placeholder="e.g., screw, bottle" className="bg-neutral-600 border-neutral-500 ..."/>
                   </div>
                   {/* File Uploader */}
                   <div>
                    <Label htmlFor="foregroundUpload" className="mb-2 block text-neutral-300 text-sm">Upload Files for Category</Label>
                    <Input id="foregroundUpload" type="file" accept=".png" multiple onChange={handleForegroundFileChange} className="bg-neutral-600 border-neutral-500 file:bg-teal-600 ..."/>
                    {currentForegroundFiles && currentForegroundFiles.length > 0 && (<p className="mt-1 text-xs text-teal-400">{currentForegroundFiles.length} file(s) selected</p>)}
                   </div>
                   {/* Add Button */}
                   <Button onClick={handleAddCategory} disabled={!currentCategoryName.trim() || !currentForegroundFiles || currentForegroundFiles.length === 0} variant="secondary" className="w-full bg-teal-600/20 ...">Add Category & Files</Button>
                   {/* Display Added Categories */}
                   {categories.length > 0 && (
                        <div className="mt-4">
                            <h5 className="text-sm font-medium mb-2 text-neutral-300">Added Categories</h5>
                            <div className="max-h-48 overflow-y-auto rounded-md border border-neutral-600 p-2 space-y-2">
                                {categories.map((category, catIndex) => (
                                    <div key={catIndex} className="text-xs">
                                        <div className="flex justify-between items-center mb-1">
                                             <Badge variant="secondary" className="bg-teal-900/50 text-teal-300 border-teal-700/50">{category.name}</Badge>
                                             <Button variant="ghost" size="sm" onClick={() => handleRemoveCategory(catIndex)} className="h-5 w-5 p-0 text-neutral-500 hover:text-red-400"><X className="h-3 w-3" /></Button>
                                        </div>
                                        <ul className="list-disc list-inside pl-2 space-y-0.5">
                                            {category.files.map((file, fileIndex) => (
                                                <li key={fileIndex} className="text-neutral-400 truncate">{file.name}</li>
                                            ))}
                                        </ul>
                                    </div>
                                ))}
                            </div>
                            <Button variant="outline" size="sm" onClick={handleClearForegrounds} className="mt-2 text-neutral-400 ...">Clear All Foregrounds</Button>
                        </div>
                   )}
                 </CardContent>
                </Card>

               {/* --- Backgrounds Card --- */}
                <Card className="bg-neutral-700 border-neutral-600">
                    <CardHeader className="pb-3"><CardTitle className="text-md font-medium flex items-center"><ImageIcon className="w-4 h-4 mr-2 text-teal-400" />Backgrounds (JPG/PNG)</CardTitle></CardHeader>
                    <CardContent className="space-y-4">
                         <div>
                             <Label htmlFor="backgroundUpload" className="mb-2 block text-neutral-300 text-sm">Upload Background Images</Label>
                             <Input id="backgroundUpload" type="file" accept=".jpg,.jpeg,.png" multiple onChange={handleBackgroundFileChange} className="bg-neutral-600 border-neutral-500 file:bg-teal-600 ..."/>
                         </div>
                         {backgroundFiles.length > 0 && (
                            <div className="mt-2">
                                <h5 className="text-sm font-medium mb-2 text-neutral-300">Selected Backgrounds ({backgroundFiles.length})</h5>
                                <div className="max-h-48 overflow-y-auto rounded-md border border-neutral-600 p-2 space-y-1">
                                    {backgroundFiles.map((file, index) => (
                                        <div key={index} className="flex justify-between items-center text-xs text-neutral-400">
                                            <span className="truncate">{file.name}</span>
                                            <Button variant="ghost" size="sm" onClick={() => handleRemoveBackgroundFile(index)} className="h-5 w-5 p-0 text-neutral-500 hover:text-red-400"><X className="h-3 w-3" /></Button>
                                        </div>
                                    ))}
                                </div>
                                <Button variant="outline" size="sm" onClick={handleClearBackgrounds} className="mt-2 text-neutral-400 ...">Clear All Backgrounds</Button>
                            </div>
                         )}
                    </CardContent>
                </Card>
             </div>

            {/* --- Augmentations Card --- */}
             <Card className="bg-neutral-700 border-neutral-600">
                <CardHeader className="pb-3"><CardTitle className="text-md font-medium flex items-center"><FileSymlink className="w-4 h-4 mr-2 text-teal-400" />Augmentations</CardTitle></CardHeader>
                <CardContent>
                     <div>
                         <Label htmlFor="augmentationUpload" className="mb-2 block text-neutral-300 text-sm">Upload Augmentation Config (Optional)</Label>
                         <Input id="augmentationUpload" type="file" accept=".yml,.yaml,.json" onChange={handleAugmentationFileChange} className="bg-neutral-600 border-neutral-500 file:bg-teal-600 ..."/>
                     </div>
                      {augmentationFile && (
                          <div className="mt-2 flex items-center justify-between bg-neutral-600/50 rounded px-2 py-1">
                              <span className="text-sm text-neutral-300 truncate">{augmentationFile.name}</span>
                              <Button variant="ghost" size="sm" onClick={handleClearAugmentation} className="h-5 w-5 p-0 text-neutral-400 hover:text-red-400"><X className="h-3 w-3" /></Button>
                          </div>
                      )}
                </CardContent>
             </Card>
           </div>

          {/* Configuration Section */}
           <div className="space-y-6">
             <h3 className="text-lg font-medium text-neutral-200 flex items-center">
               <Settings className="w-4 h-4 mr-2 text-teal-500" /> Configuration
             </h3>
             <Card className="bg-neutral-700 border-neutral-600">
               <CardContent className="pt-6 space-y-6">
                 {/* Row 1: Image Count, Max Objects */}
                 <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div><Label htmlFor="numImages" className="mb-2 block text-neutral-300 text-sm">Number of Images</Label><Input id="numImages" type="number" min={1} max={5000} value={numImages} onChange={(e) => setNumImages(Math.max(1, Number(e.target.value) || 1))} className="bg-neutral-600 ..."/></div>
                    <div><Label htmlFor="maxObjects" className="mb-2 block text-neutral-300 text-sm">Max Objects / Image</Label><Input id="maxObjects" type="number" min={1} max={25} value={maxObjects} onChange={(e) => setMaxObjects(Math.max(1, Number(e.target.value) || 1))} className="bg-neutral-600 ..."/></div>
                 </div>
                 {/* Row 2: Width, Height */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                     <div><Label htmlFor="outputWidth" className="mb-2 block text-neutral-300 text-sm">Output Width (px)</Label><Input id="outputWidth" type="number" min={64} max={4096} step={32} value={outputWidth} onChange={(e) => setOutputWidth(Math.max(64, Number(e.target.value) || 64))} className="bg-neutral-600 ..."/></div>
                     <div><Label htmlFor="outputHeight" className="mb-2 block text-neutral-300 text-sm">Output Height (px)</Label><Input id="outputHeight" type="number" min={64} max={4096} step={32} value={outputHeight} onChange={(e) => setOutputHeight(Math.max(64, Number(e.target.value) || 64))} className="bg-neutral-600 ..."/></div>
                  </div>
                 {/* Row 3: Scale Slider */}
                  <div className="space-y-2">
                    <div className="flex justify-between"><Label className="text-sm font-medium text-neutral-300">Foreground Scale Range</Label><span className="text-sm text-neutral-400">{scaleRange[0].toFixed(2)} - {scaleRange[1].toFixed(2)}</span></div>
                    <Slider min={0.05} max={1.5} step={0.05} value={scaleRange} onValueChange={(values) => { setScaleRange(values) }} className="my-2 [&>span:first-child]:h-1 [&>span:first-child>span]:bg-teal-500 [&>span:last-child]:bg-neutral-600" />
                  </div>
                 {/* Row 4: Advanced Options Accordion */}
                  <Accordion type="single" collapsible className="w-full">
                    <AccordionItem value="advanced" className="border-neutral-600"><AccordionTrigger className="text-neutral-300 hover:text-neutral-100 text-sm py-2"><div className="flex items-center"><Sliders className="w-4 h-4 mr-2" /> Advanced Options</div></AccordionTrigger><AccordionContent><div className="space-y-3 pt-3">
                        <div className="flex items-center space-x-2"><Checkbox id="avoidOverlap" checked={avoidOverlap} onCheckedChange={(checked) => setAvoidOverlap(checked === true)} className="border-neutral-500 data-[state=checked]:bg-teal-600 ..."/><Label htmlFor="avoidOverlap" className="text-neutral-300 text-sm">Avoid Object Overlap (Experimental)</Label></div>
                        <div className="flex items-center space-x-2"><Checkbox id="useParallel" checked={useParallel} onCheckedChange={(checked) => setUseParallel(checked === true)} className="border-neutral-500 data-[state=checked]:bg-teal-600 ..."/><Label htmlFor="useParallel" className="text-neutral-300 text-sm">Use Parallel Processing</Label></div>
                    </div></AccordionContent></AccordionItem>
                  </Accordion>
               </CardContent>
             </Card>
           </div>

          {/* Generate Button */}
          <Button
            onClick={handleGenerate}
            disabled={isGenerating || categories.length === 0 || backgroundFiles.length === 0}
            className="w-full md:w-auto bg-teal-600 hover:bg-teal-500 text-white disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isGenerating ? (
              <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Generating...</>
            ) : (
              <><Layers className="mr-2 h-4 w-4" />Generate Dataset</>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Status and Download Area */}
       {(status && status !== "Upload foregrounds & backgrounds, configure, then generate.") && (
        <Card className="bg-neutral-800 border-neutral-700 mt-8"> {/* Added margin */}
          <CardHeader className="pb-3 pt-4"><CardTitle className="text-lg font-light text-neutral-200">Generation Status</CardTitle></CardHeader>
          <CardContent>
             <p className={`text-sm mb-4 ${status.startsWith('❌') || status.startsWith('⚠️') ? 'text-red-400' : status.startsWith('✅') ? 'text-green-400' : 'text-neutral-300'}`}>{status}</p>
             {downloadUrl && (
              <a href={downloadUrl} download="overlay_dataset.zip">
                <Button variant="outline" className="border-teal-700 text-teal-500 hover:bg-teal-900/20 hover:text-teal-400 w-full md:w-auto">
                    <Download className="mr-2 h-4 w-4" /> Download Dataset (ZIP)
                </Button>
              </a>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}