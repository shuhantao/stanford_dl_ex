function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
f = @(s) 1 ./ (1 + exp(-s));
df = @(s) f(s) .* (1 - f(s)); 
z={};
sz = numHidden+1;
for i=1:sz
  if(i==1)  
    z{i}= (stack{i}.W*data);
  else 
    z{i}= (stack{i}.W*hAct{i-1});
  end

  z{i} = bsxfun(@plus,z{i},stack{i}.b);  
  hAct{i} = f(z{i});
end

e = exp(hAct{sz});
pred_prob = bsxfun(@rdivide,e,sum(e,1));
%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%

A = zeros(size(pred_prob));
I=sub2ind(size(A),labels', 1:size(A,2) );
A(I) = 1;
cost = -sum((A.*log(pred_prob))(:));
%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
delta={};
delta{sz+1} =  -(A-pred_prob);
for i=sz:-1:2
  delta{i} = (stack{i}.W)'*delta{i+1}.*df(z{i-1});
end
for i=1:sz
  if(i==1)
    gradStack{i}.W =delta{i+1}*data';
  else
    gradStack{i}.W =delta{i+1}*hAct{i-1}';
  end
  gradStack{i}.b = sum(delta{i+1},2);
end
%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
for i=1:sz
  gradStack{i}.W = gradStack{i}.W+ei.lambda*stack{i}.W;
  cost = cost+sum(stack{i}.W(:).^2)/2*ei.lambda;
end
%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



