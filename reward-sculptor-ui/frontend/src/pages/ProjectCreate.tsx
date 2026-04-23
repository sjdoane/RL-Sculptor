import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";

/** Legacy /projects/new route.
 *
 *  The pre-M3 project-creation form has been superseded by the library-
 *  driven flow: /library → robot card → RobotDetailModal →
 *  CreateProjectDialog (which picks the adapter + num_envs + device).
 *
 *  Anyone landing here via an old bookmark or stale link gets redirected
 *  to the library with a toast explaining the change.
 */
export default function ProjectCreate() {
  const navigate = useNavigate();
  useEffect(() => {
    toast.info("Project creation moved", {
      description: "Pick a robot from the library to scaffold a project.",
    });
    navigate("/library", { replace: true });
  }, [navigate]);
  return null;
}
