export function validateReplay(data) {
  if (!data || typeof data !== 'object' || !data.scene || !Array.isArray(data.frames)) throw new Error('文件需要包含 scene 与 frames 数组。');
  if (!['coverage','tracking'].includes(data.task)) throw new Error('回放 task 必须为 coverage 或 tracking。');
  if (data.actor && !['expert','policy','network'].includes(data.actor)) throw new Error('无法识别回放中的策略 actor。');
  let previous = -Infinity;
  for (const frame of data.frames) {
    if (!Number.isFinite(frame?.t) || frame.t < previous) throw new Error('回放时间 t 必须为递增的有限数值。');
    previous = frame.t;
    for (const key of ['uav','ship']) if (frame[key] && (!Number.isFinite(frame[key].x) || !Number.isFinite(frame[key].y))) throw new Error(`${key} 坐标必须为有限数值。`);
  }
  return {...data, actor: data.actor === 'network' ? 'policy' : data.actor || 'expert'};
}
export function frameAtTime(frames, time) {
  let low=0,high=frames.length-1;
  while(low<high){const mid=Math.ceil((low+high)/2);if(frames[mid].t<=time)low=mid;else high=mid-1;}
  return low;
}
export function formatTime(seconds) {
  const value=Math.max(0,Math.floor(Number(seconds)||0)),hours=Math.floor(value/3600),minutes=Math.floor(value/60)%60,secs=value%60;
  return `${hours?`${String(hours).padStart(2,'0')}:`:''}${String(minutes).padStart(2,'0')}:${String(secs).padStart(2,'0')}`;
}
