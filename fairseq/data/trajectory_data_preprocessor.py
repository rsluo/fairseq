import numpy as np
import os
import yaml

class TrajectoryDataPreprocessor():
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.all_filepaths = [a for (a, b, c) in os.walk(self.root_dir) if len(b) == 0]
		self.part_indices = {
								"1" : [0,1,6,7,8],
								"2" : [0,2,9,10,11],
								"3" : [0,3,12,13,14],
								"4" : [0,4,15,16,17],
								"5" : [0,5,18,19,20]
							}

	def preprocess(self, center_align = True):
		avg_dist = np.zeros(21)
		count = 0
		for path in self.all_filepaths:
			filepath = os.path.join(path, "skeleton.txt")

			# Compute average distances between joints
			with open(filepath) as file:
				count = 0.0
				frames = file.readlines()
				for frame in frames:
					count += 1
					traj = frame.split()[1:]
					skeleton = self.arrange_joints(traj)
					avg_dist += self.compute_joint_dist(skeleton)
				if count > 0:
					avg_dist /= count

		# normalize joints - refer the Moving Pose paper
		for path in self.all_filepaths:
			filepath = os.path.join(path, "skeleton.txt")
			outpath = os.path.join(path, "skeleton.norm.txt")

			norm_trajs = []
			with open(filepath) as file:
				frames = file.readlines()
				for frame in frames:
					norm_traj = np.zeros(63)
					traj = frame.split()[1:]
					joints = self.arrange_joints(traj)
					dists = self.compute_joint_dist(joints)
					norm_traj[0:3] = traj[0: 3] 
					for k, joint_idcs in self.part_indices.items():	
						for idx, joint_idx in enumerate(joint_idcs[1:]):
							norm_dist_x = (float(traj[joint_idx*3])/dists[joint_idx])*avg_dist[joint_idx]
							norm_dist_y = (float(traj[joint_idx*3+1])/dists[joint_idx])*avg_dist[joint_idx]
							norm_dist_z = (float(traj[joint_idx*3+2])/dists[joint_idx])*avg_dist[joint_idx]
							norm_traj[joint_idx*3] = float(traj[joint_idcs[idx]*3]) + norm_dist_x
							norm_traj[joint_idx*3+1] = float(traj[joint_idcs[idx]*3+1]) + norm_dist_y
							norm_traj[joint_idx*3+2] = float(traj[joint_idcs[idx]*3+2]) + norm_dist_z
					norm_trajs.append(norm_traj)
			
			if center_align:
				# center-align coordinates with wrist as origin
				aligned_trajs = []
				for traj in norm_trajs:
					aligned_traj = np.zeros(63)
					orig_x = traj[0]
					orig_y = traj[1]
					orig_z = traj[2]

					for joint_idx in range(1, 21):
						aligned_traj[joint_idx*3] = traj[joint_idx*3] - orig_x
						aligned_traj[joint_idx*3+1] = traj[joint_idx*3+1] - orig_y
						aligned_traj[joint_idx*3+2] = traj[joint_idx*3+2] - orig_z
					
					aligned_trajs.append(aligned_traj)

				norm_trajs = aligned_trajs

			with open(outpath, "w") as outfile:
				for t in norm_trajs:
					traj = " ".join([str(item) for item in t])
					outfile.write(traj)
					outfile.write("\n")
			outfile.close()

			configpath = os.path.join(self.root_dir, "skeleton.norm.yml")
			with open(configpath, "w") as outfile:
				yaml.dump(avg_dist, outfile)
			outfile.close()


	def arrange_joints(self, skeleton):
		joints = {}
		for k, val in self.part_indices.items():
			joints[k] = []
			for i in val:
				joints[k].append((skeleton[i*3], skeleton[i*3+1], skeleton[i*3+2]))
		return joints

	def compute_joint_dist(self, skeleton):
		dist = {}
		for k, joints in skeleton.items():
			dist[k] = []
			for idx in range(len(joints)-1):
				dist[k].append(np.sqrt((float(joints[idx][0]) - float(joints[idx+1][0]))**2 + \
							(float(joints[idx][1]) - float(joints[idx+1][1]))**2 + 
							(float(joints[idx][2]) - float(joints[idx+1][2]))**2))
		
		dist_vec = np.zeros(21)
		for k, joint_idx in self.part_indices.items():
			for i, idx in enumerate(joint_idx[1:]):
				dist_vec[idx] = dist[k][i]
		return dist_vec

if __name__ == '__main__':
	root_dir = '/Users/ashwini/Documents/fairseq/action_grounding/train'
	preprocessor = TrajectoryDataPreprocessor(root_dir)
	preprocessor.preprocess()