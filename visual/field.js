/* Local preference fields use +gradient: higher values attract the UAV.
   Grid row zero is south; derivatives are in preference units per metre. */
export function fieldGeometry(field) {
  const bounds = field?.bounds, values = field?.values;
  if (!Array.isArray(bounds) || bounds.length !== 4 || !bounds.every(Number.isFinite) ||
      bounds[2] <= bounds[0] || bounds[3] <= bounds[1] || !Array.isArray(values)) return null;
  const rows = values.length, cols = values[0]?.length;
  if (rows < 2 || rows > 512 || !Number.isInteger(cols) || cols < 2 || cols > 512 ||
      !values.every(row => Array.isArray(row) && row.length === cols)) return null;
  return {rows, cols, dx: (bounds[2] - bounds[0]) / (cols - 1), dy: (bounds[3] - bounds[1]) / (rows - 1),
    radius: Math.min(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 2};
}

function gradientAt(field, row, col, geometry) {
  const {rows, cols, dx, dy} = geometry;
  const valid = (r, c) => r >= 0 && r < rows && c >= 0 && c < cols &&
    Number.isFinite(field.values[r][c]) && (!field.valid || Boolean(field.valid[r]?.[c]));
  if (!Number.isInteger(row) || !Number.isInteger(col) || !valid(row, col)) return null;
  const value = field.values[row][col];
  const derivative = (r0, c0, r1, c1, spacing) => {
    const before = valid(r0, c0), after = valid(r1, c1);
    if (before && after) return (field.values[r1][c1] - field.values[r0][c0]) / (2 * spacing);
    if (after) return (field.values[r1][c1] - value) / spacing;
    if (before) return (value - field.values[r0][c0]) / spacing;
    return 0;
  };
  const x = derivative(row, col - 1, row, col + 1, dx);
  const y = derivative(row - 1, col, row + 1, col, dy);
  return {x, y, magnitude: Math.hypot(x, y)};
}

export function potentialGradient(field, row, col) {
  const geometry = fieldGeometry(field);
  return geometry ? gradientAt(field, row, col, geometry) : null;
}

export function potentialVectors(field, {scale = 1, minSpacing = 24, viewport = null} = {}) {
  const geometry = fieldGeometry(field);
  if (!geometry || !Number.isFinite(scale) || scale <= 0) return [];
  const {rows, cols, dx, dy} = geometry;
  const spacing = Number.isFinite(minSpacing) ? Math.max(1, minSpacing) : 24;
  const strideX = Math.max(1, Math.ceil(spacing / (dx * scale)));
  const strideY = Math.max(1, Math.ceil(spacing / (dy * scale)));
  // Screen-sized tiles stay anchored on the field centre as zoom changes.
  const halfX = Math.floor(strideX / 2), halfY = Math.floor(strideY / 2);
  let startX = Math.floor((cols - 1) / 2) % strideX - halfX;
  let startY = Math.floor((rows - 1) / 2) % strideY - halfY;
  if (startX > 0) startX -= strideX;
  if (startY > 0) startY -= strideY;
  const vectors = [];
  const sample = (row, col) => {
    const x = field.bounds[0] + col * dx, y = field.bounds[1] + row * dy;
    if (viewport && (x < viewport[0] || y < viewport[1] || x > viewport[2] || y > viewport[3])) return null;
    const g = gradientAt(field, row, col, geometry);
    return g && g.magnitude > 1e-12 ? {x, y, gx: g.x, gy: g.y, magnitude: g.magnitude,
      strength: Math.min(1, g.magnitude * Math.min(dx, dy) / 3),
      spacing: Math.min(strideX * dx, strideY * dy) * scale} : null;
  };
  for (let tileY = startY; tileY < rows; tileY += strideY) {
    for (let tileX = startX; tileX < cols; tileX += strideX) {
      const row0 = Math.max(0, tileY), row1 = Math.min(rows, tileY + strideY);
      const col0 = Math.max(0, tileX), col1 = Math.min(cols, tileX + strideX);
      if (viewport && (field.bounds[0] + (col1 - 1) * dx < viewport[0] ||
          field.bounds[0] + col0 * dx > viewport[2] ||
          field.bounds[1] + (row1 - 1) * dy < viewport[1] ||
          field.bounds[1] + row0 * dy > viewport[3])) continue;
      const centerRow = Math.min(rows - 1, Math.max(0, tileY + halfY));
      const centerCol = Math.min(cols - 1, Math.max(0, tileX + halfX));
      let selected = sample(centerRow, centerCol), nearest = Infinity;
      // A flat/masked tile centre must not erase a thin real field boundary.
      if (!selected) for (let row = row0; row < row1; row++) for (let col = col0; col < col1; col++) {
        const candidate = sample(row, col), distance = (row - centerRow) ** 2 + (col - centerCol) ** 2;
        if (candidate && (!selected || candidate.magnitude > selected.magnitude ||
            (candidate.magnitude === selected.magnitude && distance < nearest))) {
          selected = candidate; nearest = distance;
        }
      }
      if (selected) vectors.push(selected);
    }
  }
  return vectors;
}
