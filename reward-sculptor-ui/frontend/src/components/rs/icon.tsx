import {
  Activity, AlertCircle, AlertTriangle, ArrowLeft, ArrowRight, ArrowUpRight,
  Book, Bot, Camera, Check, CheckCircle2, ChevronDown, ChevronLeft, ChevronRight,
  ChevronUp, Circle, CircleDot, Clock, Command, Copy, Cpu, Download, ExternalLink,
  FileCode, FileText, Filter, Flag, Folder, Gauge, GitBranch, GitCommitHorizontal,
  GitCompare, History, Info, Key, Layers, LayoutGrid, Library, List, Loader2,
  Maximize2, Menu, Minus, Moon, Network, Package, PanelLeft, Pause, Pencil, Play,
  Plus, RefreshCw, ScrollText, Search, Settings, SlidersHorizontal, Sparkles,
  Square, Sun, Target, Terminal, Thermometer, TrendingUp, Trash2, Upload, User,
  Video, X, Zap, type LucideIcon,
} from "lucide-react";

/** kebab-case name → lucide component, so the ported design-prototype
 *  screens can keep their `<Icon name="file-code" />` call sites verbatim. */
const ICON_MAP: Record<string, LucideIcon> = {
  "arrow-right": ArrowRight,
  "arrow-left": ArrowLeft,
  "arrow-up-right": ArrowUpRight,
  "chevron-right": ChevronRight,
  "chevron-down": ChevronDown,
  "chevron-up": ChevronUp,
  "chevron-left": ChevronLeft,
  check: Check,
  "check-circle": CheckCircle2,
  x: X,
  plus: Plus,
  search: Search,
  terminal: Terminal,
  "git-branch": GitBranch,
  "git-commit": GitCommitHorizontal,
  "git-compare": GitCompare,
  sparkles: Sparkles,
  command: Command,
  folder: Folder,
  "file-code": FileCode,
  "file-text": FileText,
  "panel-left": PanelLeft,
  "layout-grid": LayoutGrid,
  menu: Menu,
  play: Play,
  pause: Pause,
  square: Square,
  stop: Square,
  zap: Zap,
  bot: Bot,
  settings: Settings,
  sliders: SlidersHorizontal,
  activity: Activity,
  clock: Clock,
  cpu: Cpu,
  thermometer: Thermometer,
  camera: Camera,
  video: Video,
  "refresh-cw": RefreshCw,
  upload: Upload,
  download: Download,
  copy: Copy,
  "trending-up": TrendingUp,
  "alert-triangle": AlertTriangle,
  "alert-circle": AlertCircle,
  loader: Loader2,
  circle: Circle,
  "circle-dot": CircleDot,
  book: Book,
  library: Library,
  network: Network,
  "scroll-text": ScrollText,
  filter: Filter,
  list: List,
  sun: Sun,
  moon: Moon,
  user: User,
  key: Key,
  gauge: Gauge,
  layers: Layers,
  flag: Flag,
  target: Target,
  trash: Trash2,
  pencil: Pencil,
  external: ExternalLink,
  maximize: Maximize2,
  package: Package,
  info: Info,
  minus: Minus,
  history: History,
};

export interface IconProps {
  name: string;
  size?: number;
  className?: string;
  color?: string;
  strokeWidth?: number;
}

/** Decorative by default (aria-hidden). For a standalone meaningful icon,
 *  wrap it in a labelled control (e.g. IconBtn provides aria-label). */
export function Icon({ name, size = 18, className, color, strokeWidth = 2 }: IconProps) {
  const Cmp = ICON_MAP[name] ?? Circle;
  return (
    <Cmp size={size} className={className} color={color} strokeWidth={strokeWidth} aria-hidden />
  );
}
