// Render one image for the slides inside viewer.html (run through `orca eval`; see
// tools/render_slide_images.sh). __WHICH__ is "cabinet" or a carton name; __W__/__H__ are
// the canvas size in pixels. Returns a PNG data URL on a white background.
(function () {
  const W = __W__, H = __H__, which = "__WHICH__";
  scene.background = new THREE.Color("#ffffff");
  renderer.setSize(W, H, false); camera.aspect = W / H; camera.updateProjectionMatrix();
  if (which === "cabinet") {
    mode = "cab"; buildScene();
    if (grid) grid.visible = false;
    const bb = new THREE.Box3().setFromObject(group), c = bb.getCenter(new THREE.Vector3());
    const dir = new THREE.Vector3(-0.42, -1, 0.22).normalize();
    camera.position.copy(c).add(dir.multiplyScalar(4.6)); controls.target.copy(c); camera.lookAt(c);
  } else {
    if (grid) grid.visible = false;
    while (group.children.length) group.remove(group.children[0]);
    const b = DATA.boxes.find((x) => x.name === which);
    const m = boxMesh(b);
    const a = DATA.cabinet.cartons.find((x) => x.box === which).m.slice();
    a[12] = a[13] = a[14] = 0;  // keep the shelf rotation, drop the translation
    m.matrixAutoUpdate = false; m.matrix.fromArray(a); group.add(m);
    const front = new THREE.Vector3(0, 0, 1).applyMatrix4(new THREE.Matrix4().extractRotation(m.matrix));
    const dir = new THREE.Vector3(0, -1, 0).add(front.multiplyScalar(0.75)).add(new THREE.Vector3(0, 0, 0.30)).normalize();
    camera.position.copy(dir.multiplyScalar(0.62)); controls.target.set(0, 0, 0); camera.lookAt(0, 0, 0);
  }
  renderer.render(scene, camera);
  return renderer.domElement.toDataURL("image/png");
})()
