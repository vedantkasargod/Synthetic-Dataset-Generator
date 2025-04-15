import { Sparkles } from "lucide-react"

export default function Header() {
  return (
    <header className="border-b border-neutral-800 bg-neutral-900 py-6">
      <div className="container mx-auto px-4 max-w-6xl">
        <div className="flex items-center justify-center">
          <Sparkles className="w-6 h-6 mr-2 text-teal-500" />
          <h1 className="text-2xl md:text-3xl font-light tracking-wide text-center text-neutral-100">
            AI Image Studio
          </h1>
        </div>
      </div>
    </header>
  )
}
