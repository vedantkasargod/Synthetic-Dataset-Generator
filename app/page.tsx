"use client"
import Header from "@/components/header"
import PromptGeneration from "@/components/prompt-generation"
import OverlayGeneration from "@/components/overlay-generation"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function Home() {
  return (
    <div className="min-h-screen bg-neutral-900 text-neutral-200">
      <Header />
      <main className="container mx-auto px-4 py-8 max-w-6xl">
        <Tabs defaultValue="prompt" className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-8 bg-neutral-800 p-1 rounded-lg">
            <TabsTrigger
              value="prompt"
              className="rounded-md py-3 data-[state=active]:bg-teal-500 data-[state=active]:text-neutral-900 data-[state=active]:shadow transition-all"
            >
              Prompt Generation
            </TabsTrigger>
            <TabsTrigger
              value="overlay"
              className="rounded-md py-3 data-[state=active]:bg-teal-500 data-[state=active]:text-neutral-900 data-[state=active]:shadow transition-all"
            >
              Overlay Generation
            </TabsTrigger>
          </TabsList>
          <TabsContent value="prompt">
            <PromptGeneration />
          </TabsContent>
          <TabsContent value="overlay">
            <OverlayGeneration />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
