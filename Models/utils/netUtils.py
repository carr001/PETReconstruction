import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor

########################### code borror from DIP ##################################
def get_kernel(factor, kernel_type, phase, kernel_width, support=None, sigma=None):
	assert kernel_type in ['lanczos', 'gauss', 'box']

	# factor  = float(factor)
	if phase == 0.5 and kernel_type != 'box':
		kernel = np.zeros([kernel_width - 1, kernel_width - 1])
	else:
		kernel = np.zeros([kernel_width, kernel_width])

	if kernel_type == 'box':
		assert phase == 0.5, 'Box filter is always half-phased'
		kernel[:] = 1. / (kernel_width * kernel_width)

	elif kernel_type == 'gauss':
		assert sigma, 'sigma is not specified'
		assert phase != 0.5, 'phase 1/2 for gauss not implemented'

		center = (kernel_width + 1.) / 2.
		print(center, kernel_width)
		sigma_sq = sigma * sigma

		for i in range(1, kernel.shape[0] + 1):
			for j in range(1, kernel.shape[1] + 1):
				di = (i - center) / 2.
				dj = (j - center) / 2.
				kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj) / (2 * sigma_sq))
				kernel[i - 1][j - 1] = kernel[i - 1][j - 1] / (2. * np.pi * sigma_sq)
	elif kernel_type == 'lanczos':
		assert support, 'support is not specified'
		center = (kernel_width + 1) / 2.

		for i in range(1, kernel.shape[0] + 1):
			for j in range(1, kernel.shape[1] + 1):

				if phase == 0.5:
					di = abs(i + 0.5 - center) / factor
					dj = abs(j + 0.5 - center) / factor
				else:
					di = abs(i - center) / factor
					dj = abs(j - center) / factor

###################################### code borrow from https://github.com/nikopj/DGCN.git #############
### knn.py
def windowedTopK(h, K, M, mask):
	""" Returns top K feature vector indices for
	h: (B, C, H, W) input feature
	M: window side-length
	mask: (H*W, H*W) Graph mask.
	output: (B, K, H, W) K edge indices (of flattened image) for each pixel
	"""
	# stack image windows
	hs = stack(h, M)          # (B,I,J,C,M,M)
	I, J = hs.shape[1], hs.shape[2]
	# move stack to match dimension to build batched Graph Adjacency matrices
	hbs = batch_stack(hs)     # (B*I*J,C,M,M)
	G = graphAdj(hbs, mask)         # (B*I*J, M*M, M*M)
	# find topK in each window, unbatch the stack, translate window-index to tile index
	# (B*I*J,M*M,K) -> (B*I*J,K,M*M) -> (B*I*J, K, M, M)
	edge = torch.topk(G, K, largest=False).indices.permute(0,2,1).reshape(-1,K,M,M)
	edge = unbatch_stack(edge, (I,J)) # (B,I,J,K,M,M)
	return indexTranslate(edge) # (B,K,H,W)

def localTopK(h, K, M, mask):
	B, C, H, W = h.shape
	m1, m2 = np.floor((M-1)/2), np.ceil((M-1)/2)
	edge = torch.empty(B,K,H,W)
	for i in range(H): # (p,q) indexes the top-left corner of the window
		p = int(np.clip(i-m1,0,H-M))
		for j in range(W):
			q = int(np.clip(j-m1,0,W-M))
			loc_window = h[:,:,p:p+M,q:q+M]
			v = h[:,:,i,j][...,None,None]
			dist = torch.sum((loc_window - v)**2, dim=1).reshape(-1,M*M)
			mi, mj = i-p, j-q              # index in local window coords
			# window coords local area
			mi = np.clip(np.arange(mi-1,mi+2),0,M-1).astype(np.int64)
			mj = np.clip(np.arange(mj-1,mj+2),0,M-1).astype(np.int64)
			mask = mi.reshape(-1,1)*M + mj.reshape(1,-1)               # window flattened idx local area
			dist[:,mask] = torch.tensor(np.inf)
			loc_edge = torch.topk(dist, K, largest=False).indices
			m, n = loc_edge//M, loc_edge%M # flat window index to coord
			m, n = m+p, n+q                # window coord to image coord
			edge[:,:,i,j] = W*m + n        # image coord to flat image index
	return edge.long()

def graphAdj(h, mask):
	""" ||h_j - h_i||^2 L2 similarity matrix formation
	Using the following identity:
		||h_j - h_i||^2 = ||h_j||^2 - 2h_j^Th_i + ||h_i||^2
	h: input (B, C, H, W)
	mask: (H*W, H*W)
	G: output (B, N, N), N=H*W
	"""
	N = h.shape[2]*h.shape[3] # num pixels
	v = h.reshape(-1,h.shape[1],N) # (B, C, N)
	vtv = torch.bmm(v.transpose(1,2), v) # batch matmul, (B, N, N)
	normsq_v = vtv.diagonal(dim1=1, dim2=2) # (B, N)
	# need to add normsq_v twice, with broadcasting in row/col dim
	G = normsq_v.unsqueeze(1) - 2*vtv + normsq_v.unsqueeze(2) # (B, N, N)
	# apply local mask (local distances set to infinity)
	G[:,~mask] = torch.tensor(np.inf)
	return G

def localMask(H,W,M):
	""" generate adjacency matrix mask to exclude local area around pixel
	H: image height
	W: image width
	M: local area side-length (square filter side-length)
	output: (H*W, H*W)
	"""
	N = H*W
	mask = torch.ones(N,N, dtype=torch.bool)
	m = (M-1)//2
	for pixel in range(N):
		for delta_row in range(-m,m+1):
			# absolute row number
			row = int(np.floor(pixel/W)) + delta_row
			# don't exit image
			if row < 0 or row > H-1:
				continue
			# clip local area to stop wrap-around
			a1 = int(np.clip(pixel%W - m, 0, W-1))
			a2 = int(np.clip(pixel%W + m, 0, W-1))
			local_row = np.arange(row*W + a1, row*W + a2 + 1) # local area of row
			mask[pixel, local_row] = False
	return mask

def getLabelVertex(input, edge):
	""" Return edge labels and verticies for each pixel in input, derived from edges.
	Edges correspond to K-Regular Graph.
	input: (B, C, H, W)
	edge: input (B, K, H, W), edge indices
	label, vertex_set: output (B, K, N, C)
	"""
	B, K, H, W = edge.shape
	C, N = input.shape[1], H*W
	v = input.reshape(B, C, N)
	edge = edge.reshape(B, K, N)
	# differentiate indices in the batch dimension,
	edge = edge + torch.arange(0,B,device=input.device).reshape(-1,1,1)*N
	# put pixels in batch dimension
	v  = v.permute(0,2,1).reshape(-1, C)          # (BN, C)
	vS = torch.index_select(v, 0, edge.flatten()) # (BKN, C)
	# correspond pixels to nonlocal neighbors
	v  = v.reshape(B, N, C)
	vS = vS.reshape(B, K, N, C)
	label = vS - v.unsqueeze(1) # (B, K, N, C)
	return label, vS

def receptiveField(coord, edge, depth):
	""" Return receptive field / neighbors pixel at specified coordinate
	after iterating with depth > 1.
	coord: 2-tuple or flattened index of spatial dimension
	edge: (1,K,H,W) KNN list for each pixel in (H,W)
	depth: scalar for depth of network considering these connections
	"""
	m, n = coord
	neighbors = edge[0,:,m,n]
	if depth < 1:
		return neighbors
	cols = edge.shape[-1]
	for i in range(len(neighbors)):
		new_coord = (neighbors[i]//cols, neighbors[i]%cols)
		neighbors = torch.cat([neighbors, receptiveField(new_coord, edge, depth-1)])
	return neighbors

def imgLoad(path, gray=False):
	""" Load batched tensor image (1,C,H,W) from file path.
	"""
	if gray:
		return to_tensor(Image.open(path).convert('L'))[None,...]
	return to_tensor(Image.open(path))[None,...]

def awgn(input, noise_std):
	""" Additive White Gaussian Noise
	y: clean input image
	noise_std: (tuple) noise_std of batch size N is uniformly sampled
	           between noise_std[0] and noise_std[1]. Expected to be in interval
			   [0,255]
	"""
	if not isinstance(noise_std, (list, tuple)):
		sigma = noise_std
	else: # uniform sampling of sigma
		sigma = noise_std[0] + \
		       (noise_std[1] - noise_std[0])*torch.rand(len(input),1,1,1, device=input.device)
	return input + torch.randn_like(input) * (sigma/255)

def pre_process(x, window_size):
	params = []
	# mean-subtract
	xmean = x.mean(dim=(2,3), keepdim=True)
	x = x - xmean
	params.append(xmean)
	# pad signal for windowed processing (in GraphConv)
	pad = calcPad2D(*x.shape[2:], window_size)
	x = F.pad(x, pad, mode='reflect')
	params.append(pad)
	return x, params

def post_process(x, params):
	# unpad
	pad = params.pop()
	x = unpad(x, pad)
	# add mean
	xmean = params.pop()
	x = x + xmean
	return x

def calcPad1D(L, M):
	""" Return symmetric pad sizes for length L 1D signal
	to be divided into non-overlapping windows of size M.
	"""
	if L%M == 0:
		Lpad = [0,0]
	else:
		Lprime = np.ceil(L/M) * M
		Ldiff  = Lprime - L
		Lpad   = [int(np.floor(Ldiff/2)), int(np.ceil(Ldiff/2))]
	return Lpad

def calcPad2D(H, W, M):
	""" Return pad sizes for image (H,W) to be
	divided into windows of size (MxM).
	(H,W): input height, width
	M: window size
	output: (padding_left, padding_right, padding_top, padding_bottom)
	"""
	return (*calcPad1D(W,M), *calcPad1D(H,M))

def unpad(I, pad):
	""" Remove padding from 2D signal I.
	"""
	if pad[3] == 0 and pad[1] > 0:
		return I[..., pad[2]:, pad[0]:-pad[1]]
	elif pad[3] > 0 and pad[1] == 0:
		return I[..., pad[2]:-pad[3], pad[0]:]
	elif pad[3] == 0 and pad[1] == 0:
		return I[..., pad[2]:, pad[0]:]
	else:
		return I[..., pad[2]:-pad[3], pad[0]:-pad[1]]

def stack(T, M):
	""" Stack I (B, C, H, W) into patches of size (MxM).
	output: (B, I, J, C, H, W).
	"""
	# (B,C,H,W) -> unfold (B,C,I,J,M,M) -> permute (B,I,J,C,M,M)
	return T.unfold(2,M,M).unfold(3,M,M).permute(0,2,3,1,4,5)

def batch_stack(S):
	""" Reorder stack (B, I, J, C, M, M) so that
	patches are stacked in the batch dimension,
	output: (B*I*J, C, H, W)
	"""
	C, M = S.shape[3], S.shape[-1]
	return S.reshape(-1,C,M,M)

def unbatch_stack(S, grid_shape):
	""" Reorder batched stack into non-batcheys)
	(B*I*J, C, M, M) -> (B, I, J, C, M, M)
	"""
	I, J = grid_shape
	C, M = S.shape[1], S.shape[2]
	return S.reshape(-1, I, J, C, M, M)

def unstack(S):
	""" Tile patches to form image
	(B, I, J, C, M, M) -> (B, C, I*M, J*M)
	"""
	B, I, J, C, M, _ = S.shape
	T = S.reshape(B, I*J, C*M*M).permute(0,2,1)
	return F.fold(T, (I*M, J*M), M, stride=M)

def indexTranslate(idx):
	""" Translate stacked grid (flattened MxM window) index (B,I,J,K,M,M)
	to tiled-image (flattened HxW) index, (B,K,H,W)
	"""
	B, I, J, K, M, _ = idx.shape
	# each idx entries grid-index
	grid_idx = torch.arange(0,I*J,device=idx.device).repeat_interleave(M*M).reshape(1,I,J,1,M,M).repeat_interleave(K, dim=3)
	# grid index row and column (inter-window)
	gi, gj = grid_idx//J, grid_idx%J
	# window index row and column (intra-window)
	wi, wj = idx//M, idx%M
	# global index row and column
	m, n = wi+gi*M, wj+gj*M
	# global flattened index
	p = J*M*m + n
	# stack to tile (unstack requires float)
	return unstack(p.float()).long()

def powerMethod(A, b, num_iter=1000, tol=1e-6, verbose=True):
	eig_old = torch.zeros(1)
	flag_tol_reached = False
	for it in range(num_iter):
		b = A(b)
		b = b / torch.norm(b)
		eig_max = torch.sum(b*A(b))
		if verbose:
			print('i:{0:3d} \t |e_new - e_old|:{1:2.2e}'.format(it,abs(eig_max-eig_old).item()))
		if abs(eig_max-eig_old)<tol:
			flag_tol_reached = True
			break
		eig_old = eig_max
	if verbose:
		print('tolerance reached!',it)
		print(f"L = {eig_max.item():.3e}")
	return eig_max.item(), b, flag_tol_reached
