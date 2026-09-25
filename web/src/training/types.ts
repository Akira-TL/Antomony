export type Phase='motor'|'meta'|'autonomous';
export type GroupId='action'|'gate'|'query'|'key'|'rate';
export type Lesson='straight'|'turn'|'random';
export interface GroupState {id:GroupId;label:string;start:number;end:number;frozen:boolean;manual:boolean;reason:string}
export interface EpisodeRecord {episode:number;phase:Phase;lesson:Lesson;steps:number;reward:number;reached:boolean;reason:string;self_updates:number;outer_updates:number;checkpoint:string}
export interface WeightTracePoint {sequence:number;tick:number;episode:number;phase:Phase;source:'initial'|'skip'|'self'|'outer';status:string;value:number;delta:number;changed:number;total_change:number}
export interface WeightTrace {session:string;row:number;column:number;points:WeightTracePoint[]}
export interface TrainingState {
  session:string;paused:boolean;error:string;phase:Phase;lesson:Lesson;speed:number;tick:number;episode:number;steps:number;horizon:number;
  x:number;y:number;heading:number;target_x:number;target_y:number;distance:number;max_turn:number;
  move:boolean;turn:number;move_probability:number;write_probability:number;write_status:string;
  reward:number;total_reward:number;self_updates:number;outer_updates:number;
  inputs:string[];observation:number[];groups:GroupState[];trainable?:boolean[][];weights:number[][];initial:number[][];self_delta:number[][];outer_delta:number[][];history:EpisodeRecord[];
}
export interface TrainingCommand {
  action:'play'|'pause'|'step'|'phase'|'lesson'|'freeze'|'freeze_all'|'speed'|'turn'|'reset';
  phase?:Phase;lesson?:Lesson;group?:GroupId;frozen?:boolean;speed?:1|4|16;max_turn?:number;
}

export type RecurrentPhase='motor'|'memory'|'adaptive'|'autonomous';
export type RecurrentTask='normal'|'shift'|'sensor'|'mixed';
export type PerturbationKind='none'|'turn'|'sensor';
export type WriteMode='off'|'learned'|'always';
export interface RecurrentGroup {id:string;label:string;values:number[][];changes:number[][];trainable:boolean[][]}
export interface RecurrentEpisode {episode:number;phase:RecurrentPhase;task:RecurrentTask;perturbation_kind:PerturbationKind;perturbation:number;reached:boolean;steps:number;reward:number;writes:number;checkpoint:string}
export interface RecurrentState {
  session:string;paused:boolean;error:string;phase:RecurrentPhase;task:RecurrentTask;speed:number;tick:number;episode:number;steps:number;horizon:number;
  x:number;y:number;heading:number;target_x:number;target_y:number;distance:number;max_turn:number;
  move:boolean;turn:number;move_probability:number;reward:number;total_reward:number;reached:boolean;
  perturbation:number;perturbation_kind:PerturbationKind;active_perturbation:number;outer_updates:number;self_updates:number;write_mode:WriteMode;write_probability:number;write_status:string;
  hidden:number[];fast:number[];fast_delta:number[];groups:RecurrentGroup[];history:RecurrentEpisode[];
}
export interface RecurrentCommand {action:'play'|'pause'|'step'|'phase'|'task'|'speed'|'reset'|'write_mode';phase?:RecurrentPhase;task?:RecurrentTask;speed?:1|4|16;write_mode?:WriteMode}
