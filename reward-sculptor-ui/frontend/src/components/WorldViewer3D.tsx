/** Interactive 3D view of a compiled world scene graph.
 *
 *  Renders the EXACT compiled scene (promoted selection or draft
 *  dry-run) served by the backend scene exporter — never a client-side
 *  re-derivation of the spec. Orbit/pan/zoom via OrbitControls, click
 *  to select a semantic entity (course element, object, zone, robot,
 *  terrain); selection is controlled by the parent so an inspector
 *  panel can drive and reflect it.
 *
 *  Conventions: MuJoCo is Z-up, so the camera's `up` is +Z and the
 *  ground grid lies in the XY plane. Geom groups ≥ 3 (collision
 *  shells) are hidden, matching MuJoCo's default visualizer. Gap
 *  markers and zone sites render translucent — spacing and predicate
 *  regions, not solids.
 */
import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type { WorldScene, WorldSceneGeom } from "@/lib/types";

const HIGHLIGHT = new THREE.Color(0xffa94d);

function quatToThree(q: number[]): THREE.Quaternion {
  // MuJoCo order [w, x, y, z] → three [x, y, z, w].
  return new THREE.Quaternion(q[1], q[2], q[3], q[0]);
}

function geomGeometry(
  geom: WorldSceneGeom, scene: WorldScene,
): THREE.BufferGeometry | null {
  const s = geom.size;
  switch (geom.type) {
    case "box":
      return new THREE.BoxGeometry(s[0] * 2, s[1] * 2, s[2] * 2);
    case "sphere":
      return new THREE.SphereGeometry(s[0], 24, 16);
    case "ellipsoid": {
      const g = new THREE.SphereGeometry(1, 24, 16);
      g.scale(s[0], s[1], s[2]);
      return g;
    }
    case "cylinder": {
      // MuJoCo cylinders/capsules: size = [radius, half-length, _], axis
      // along +Z; three's primitives run along +Y, so rotate into Z.
      const g = new THREE.CylinderGeometry(s[0], s[0], s[1] * 2, 24);
      g.rotateX(Math.PI / 2);
      return g;
    }
    case "capsule": {
      // three's length param is the cylindrical mid-section (2 × half).
      const g = new THREE.CapsuleGeometry(s[0], s[1] * 2, 6, 16);
      g.rotateX(Math.PI / 2);
      return g;
    }
    case "mesh": {
      const mesh = geom.mesh ? scene.meshes[geom.mesh] : undefined;
      if (!mesh) return null;
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(
        mesh.vertices, 3));
      g.setIndex(mesh.faces);
      g.computeVertexNormals();
      return g;
    }
    case "hfield": {
      const hf = geom.hfield ? scene.hfields[geom.hfield] : undefined;
      if (!hf) return null;
      const [rx, ry, ez] = hf.size;
      const g = new THREE.PlaneGeometry(
        rx * 2, ry * 2, hf.ncol - 1, hf.nrow - 1);
      const pos = g.getAttribute("position");
      for (let row = 0; row < hf.nrow; row++) {
        for (let col = 0; col < hf.ncol; col++) {
          // PlaneGeometry vertices run row-major from +y to -y; hfield
          // data is row-major from -y to +y.
          const v = (hf.nrow - 1 - row) * hf.ncol + col;
          pos.setZ(v, hf.data[row * hf.ncol + col] * ez);
        }
      }
      g.computeVertexNormals();
      return g;
    }
    case "plane":
      // Rendered as the ground grid; no solid geometry.
      return null;
    default:
      return null;
  }
}

export default function WorldViewer3D({
  scene, selectedEntity, onSelectEntity, height = 420,
}: {
  scene: WorldScene;
  selectedEntity: string | null;
  onSelectEntity: (id: string | null) => void;
  height?: number;
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  // Entity-id → materials, for highlight without a rebuild.
  const materialsRef = useRef<Map<string, THREE.MeshStandardMaterial[]>>(
    new Map());
  const baseColorRef = useRef<Map<THREE.MeshStandardMaterial, THREE.Color>>(
    new Map());
  const onSelectRef = useRef(onSelectEntity);
  onSelectRef.current = onSelectEntity;

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const width = mount.clientWidth || 800;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    mount.appendChild(renderer.domElement);

    const world = new THREE.Scene();
    world.background = new THREE.Color(0x11151c);

    const camera = new THREE.PerspectiveCamera(
      45, width / height, 0.01, 300);
    camera.up.set(0, 0, 1);

    const hemi = new THREE.HemisphereLight(0xffffff, 0x445566, 1.1);
    world.add(hemi);
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(4, -6, 8);
    world.add(sun);

    // GridHelper is LineSegments (not Mesh) — dispose it explicitly in
    // cleanup; the Mesh-only traverse below won't reach it.
    const grid = new THREE.GridHelper(60, 60, 0x2f3a4a, 0x222b38);
    grid.rotation.x = Math.PI / 2; // GridHelper is XZ; rotate into XY
    world.add(grid);

    const materials = materialsRef.current;
    const baseColors = baseColorRef.current;
    materials.clear();
    baseColors.clear();
    const pickables: THREE.Mesh[] = [];
    const bbox = new THREE.Box3();

    const register = (
      entityId: string, mat: THREE.MeshStandardMaterial,
    ) => {
      if (!materials.has(entityId)) materials.set(entityId, []);
      materials.get(entityId)!.push(mat);
      baseColors.set(mat, mat.color.clone());
    };

    for (const geom of scene.geoms) {
      if (geom.group >= 3) continue; // collision shells
      const geometry = geomGeometry(geom, scene);
      if (!geometry) continue;
      const [r, g, b, a] = geom.rgba;
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(r, g, b),
        roughness: 0.85,
        metalness: 0.05,
        transparent: a < 1,
        opacity: a,
      });
      const mesh = new THREE.Mesh(geometry, mat);
      mesh.position.set(geom.pos[0], geom.pos[1], geom.pos[2]);
      mesh.quaternion.copy(quatToThree(geom.quat));
      mesh.userData.entityId = geom.entity_id;
      world.add(mesh);
      pickables.push(mesh);
      register(geom.entity_id, mat);
      if (geom.type !== "plane") bbox.expandByObject(mesh);
    }

    for (const site of scene.sites) {
      const s = site.size;
      const geometry = site.type === "sphere"
        ? new THREE.SphereGeometry(s[0], 20, 14)
        : new THREE.BoxGeometry(s[0] * 2, s[1] * 2, s[2] * 2);
      const [r, g, b] = site.rgba;
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(r, g, b),
        transparent: true,
        opacity: 0.3,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(geometry, mat);
      mesh.position.set(site.pos[0], site.pos[1], site.pos[2]);
      mesh.quaternion.copy(quatToThree(site.quat));
      mesh.userData.entityId = site.entity_id;
      world.add(mesh);
      pickables.push(mesh);
      register(site.entity_id, mat);
      bbox.expandByObject(mesh);
    }

    // Gap markers: geometry-less course elements rendered as translucent
    // spacing slabs so "5 elements / 3 boxes" is visibly coherent.
    for (const entity of scene.entities) {
      if (!entity.marker) continue;
      const { pos, size } = entity.marker;
      const mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(0xd9822b),
        transparent: true,
        opacity: 0.22,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(size[0] * 2, size[1] * 2, size[2] * 2), mat);
      mesh.position.set(pos[0], pos[1], pos[2]);
      mesh.userData.entityId = entity.id;
      world.add(mesh);
      pickables.push(mesh);
      register(entity.id, mat);
    }

    if (bbox.isEmpty()) bbox.setFromCenterAndSize(
      new THREE.Vector3(0, 0, 0.5), new THREE.Vector3(2, 2, 1));
    const center = bbox.getCenter(new THREE.Vector3());
    const span = Math.max(bbox.getSize(new THREE.Vector3()).length(), 1.5);
    camera.position.set(
      center.x + span * 0.8, center.y - span * 0.9, center.z + span * 0.65);
    camera.lookAt(center);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.copy(center);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.maxDistance = span * 8;
    controls.update();

    // Click-to-select with a small drag tolerance so orbiting never
    // triggers selection.
    const raycaster = new THREE.Raycaster();
    const down = new THREE.Vector2();
    const onPointerDown = (event: PointerEvent) => {
      down.set(event.clientX, event.clientY);
    };
    const onPointerUp = (event: PointerEvent) => {
      if (Math.hypot(event.clientX - down.x, event.clientY - down.y) > 5) {
        return;
      }
      const rect = renderer.domElement.getBoundingClientRect();
      const ndc = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1);
      raycaster.setFromCamera(ndc, camera);
      const hits = raycaster.intersectObjects(pickables, false);
      onSelectRef.current(
        hits.length > 0
          ? String(hits[0].object.userData.entityId)
          : null);
    };
    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointerup", onPointerUp);

    let frame = 0;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      controls.update();
      renderer.render(world, camera);
    };
    animate();

    const onResize = () => {
      const w = mount.clientWidth || width;
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
      renderer.setSize(w, height);
    };
    const observer = new ResizeObserver(onResize);
    observer.observe(mount);

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      controls.dispose();
      world.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          obj.geometry.dispose();
          (Array.isArray(obj.material)
            ? obj.material : [obj.material]).forEach((m) => m.dispose());
        }
      });
      grid.geometry.dispose();
      (Array.isArray(grid.material)
        ? grid.material : [grid.material]).forEach((m) => m.dispose());
      renderer.dispose();
      // Repeated builder previews create fresh WebGL contexts; force the
      // old one down instead of waiting for GC (browsers cap ~16 live).
      renderer.forceContextLoss();
      mount.removeChild(renderer.domElement);
      materials.clear();
      baseColors.clear();
    };
    // Rebuild only when the scene payload itself changes.
  }, [scene, height]);

  // Highlight pass — no scene rebuild on selection change. `scene` is a
  // dep so a rebuilt payload re-applies the current highlight to the
  // fresh materials instead of rendering unselected.
  useEffect(() => {
    for (const [entityId, mats] of materialsRef.current) {
      const active = entityId === selectedEntity;
      for (const mat of mats) {
        const base = baseColorRef.current.get(mat);
        if (!base) continue;
        mat.emissive = active ? HIGHLIGHT : new THREE.Color(0x000000);
        mat.emissiveIntensity = active ? 0.45 : 0;
        mat.color.copy(active ? base.clone().lerp(HIGHLIGHT, 0.25) : base);
      }
    }
  }, [selectedEntity, scene]);

  return (
    <div
      ref={mountRef}
      style={{ width: "100%", height, borderRadius: 8, overflow: "hidden",
               cursor: "pointer" }}
      aria-label="Interactive 3D world scene"
    />
  );
}
