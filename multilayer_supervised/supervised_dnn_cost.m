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
%fprintf('numHidden = %d\n',numHidden);
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
numSamples = size(data,2);
%% forward prop
%%% YOUR CODE HERE %%%
sz = numHidden+1;
for l = 1:numHidden,
    if(l==1)
        hAct{1} = stack{1}.W * data + repmat(stack{1}.b,1,numSamples);
    else 
        hAct{l} = stack{l}.W * hAct{l-1} + repmat(stack{l}.b,1,numSamples);
    end
    hAct{l} = 1 ./(1 + exp(-hAct{l}));
end;
hAct{sz} = stack{sz}.W * hAct{sz-1} + repmat(stack{sz}.b,1,numSamples);
e = exp(hAct{sz});
sm = sum(e,1);
hAct{sz} = bsxfun(@rdivide,e,sm);
pred_prob = hAct{sz};
%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
e = log(hAct{sz});
I = sub2ind(size(e),labels',1:numSamples);
cost = -sum(e(I));
%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
delta={};
A = zeros(size(hAct{sz})); 
A(I) = 1; 
delta{sz} = -(A - hAct{sz});
gradStack{sz}.W = delta{sz} * hAct{sz-1}';
gradStack{sz}.b = sum(delta{sz},2);
for l = numHidden:-1:1,
    df = hAct{l} .* (1 - hAct{l});
    delta{l} = (stack{l+1}.W' * delta{l+1}) .* df;
    if (l==1)
         gradStack{l}.W = delta{l} * data';
    else
        gradStack{l}.W = delta{l} * hAct{l-1}';
    end
    gradStack{l}.b = sum(delta{l},2);
end;

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
wcost = 0;
for l = 1:sz,
    wcost = wcost + 0.5 * ei.lambda * sum(stack{l}.W(:) .^ 2 );
end;
cost = cost ./ numSamples + wcost;
%cost = cost ./ numSamples;
for  l = 1:numHidden+1,
    gradStack{l}.b = gradStack{l}.b / numSamples;
    gradStack{l}.W = gradStack{l}.W / numSamples;
    gradStack{l}.W = gradStack{l}.W + ei.lambda * stack{l}.W;
end;


%% reshape gradients into vector
[grad] = stack2params(gradStack);
%size(grad)
end